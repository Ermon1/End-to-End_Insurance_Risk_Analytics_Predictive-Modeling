# src/models/task_4/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
from src.utility.config_loader import loader as config_loader
from pathlib import Path
import re

# Load config
config = config_loader.load('data.yaml')
ARTIFACT_DIR = Path(config['data']['artifact_dir'])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class DataPreparer:
    def __init__(
        self, 
        df: pd.DataFrame, 
        target: str = "TotalClaims", 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        self.df = df.copy()
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        # Columns with extreme missing values to drop
        self.drop_cols = [
            "CustomValueEstimate", "WrittenOff", "Rebuilt", "Converted",
            "CrossBorder", "NumberOfVehiclesInFleet", "TransactionMonth","VehicleIntroDate"
        ]

        # Columns to impute
        self.numeric_cols = [
            "CapitalOutstanding", "Cylinders", "cubiccapacity",
            "kilowatts", "NumberOfDoors"
        ]
        self.categorical_cols = [
            "Bank", "AccountType", "MaritalStatus", "Gender", "mmcode",
            "VehicleType", "make", "Model", "bodytype",
            "VehicleIntroDate", "NewVehicle"
        ]

    def drop_extreme_missing(self):
        """Drop columns with extreme missing values."""
        # Filter columns that actually exist in the dataframe
        existing_drop_cols = [c for c in self.drop_cols if c in self.df.columns]
        print(f"Dropping columns: {existing_drop_cols}")
        self.df.drop(columns=existing_drop_cols, inplace=True, errors='ignore')

    def clean_string_to_numeric(self, series):
        """Convert a series with mixed strings/numbers to numeric."""
        def convert_value(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float, np.number)):
                return float(x)
            
            # Convert to string and clean
            x_str = str(x).strip()
            if x_str == '' or x_str.lower() in ['nan', 'none', 'null', 'na']:
                return np.nan
            
            # Remove thousand separators (both . and ,)
            x_str = re.sub(r'[.,](?=\d{3})', '', x_str)
            # Convert decimal comma to dot
            x_str = x_str.replace(',', '.')
            # Remove any remaining non-numeric characters except decimal point and minus
            x_str = re.sub(r'[^\d\.\-]', '', x_str)
            
            if x_str == '' or x_str == '-':
                return np.nan
            
            try:
                return float(x_str)
            except:
                return np.nan
        
        return series.apply(convert_value)

    def clean_numeric(self):
        """Clean numeric columns: handle comma decimals, spaces, empty strings."""
        print("Cleaning numeric columns...")
        for col in self.numeric_cols:
            if col in self.df.columns:
                print(f"  Processing {col}...")
                # Get unique values before cleaning for debugging
                unique_before = self.df[col].dropna().unique()[:10]
                print(f"    Sample values before: {unique_before}")
                
                # Clean the column
                self.df[col] = self.clean_string_to_numeric(self.df[col])
                
                # Get unique values after cleaning
                unique_after = self.df[col].dropna().unique()[:10]
                print(f"    Sample values after: {unique_after}")
                
                # Count NaN values
                nan_count = self.df[col].isna().sum()
                print(f"    NaN values: {nan_count}")

    def clean_categorical(self):
        """Clean categorical columns: strip spaces and replace empty strings with 'Unknown'."""
        print("Cleaning categorical columns...")
        for col in self.categorical_cols:
            if col in self.df.columns:
                # Convert to string, handle NaN
                self.df[col] = self.df[col].fillna('Unknown').astype(str).str.strip()
                # Replace empty strings
                self.df[col] = self.df[col].replace(['', 'nan', 'NaN', 'None', 'null'], 'Unknown')
                
                # Print summary
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique values")

    def impute_missing(self):
        """Impute missing values in numeric and categorical columns."""
        print("Imputing missing values...")
        
        # First clean the data
        self.clean_numeric()
        self.clean_categorical()
        
        # Check which columns actually exist
        existing_numeric = [c for c in self.numeric_cols if c in self.df.columns]
        existing_categorical = [c for c in self.categorical_cols if c in self.df.columns]
        
        print(f"  Numeric columns to impute: {existing_numeric}")
        print(f"  Categorical columns to impute: {existing_categorical}")
        
        # Numeric imputer
        if existing_numeric:
            # Check for NaN values before imputation
            nan_before = self.df[existing_numeric].isna().sum().sum()
            print(f"  Total NaN in numeric columns before imputation: {nan_before}")
            
            num_imputer = SimpleImputer(strategy="median")
            self.df[existing_numeric] = num_imputer.fit_transform(self.df[existing_numeric])
            joblib.dump(num_imputer, ARTIFACT_DIR / "numeric_imputer.pkl")
            print("  Numeric imputation complete")
        
        # Categorical imputer
        if existing_categorical:
            # Check for NaN values before imputation
            nan_before = self.df[existing_categorical].isna().sum().sum()
            print(f"  Total NaN in categorical columns before imputation: {nan_before}")
            
            cat_imputer = SimpleImputer(strategy="most_frequent")
            self.df[existing_categorical] = cat_imputer.fit_transform(self.df[existing_categorical])
            joblib.dump(cat_imputer, ARTIFACT_DIR / "categorical_imputer.pkl")
            print("  Categorical imputation complete")

    def create_model_features(self):
        """Generate target variables for different models."""
        print("Creating model features...")
        df = self.df.copy()
        
        # Ensure target column exists
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe")
        
        # Print target column info
        print(f"  Target column: {self.target}")
        print(f"  Target dtype: {df[self.target].dtype}")
        print(f"  Target sample values: {df[self.target].head().tolist()}")
        
        # Claim probability (classification)
        df["has_claim"] = (df[self.target] > 0).astype(int)
        print(f"  Claim frequency: {df['has_claim'].mean():.2%}")
        
        # Claim severity (regression) - only policies with claims
        df_severity = df[df[self.target] > 0].copy()
        print(f"  Severity dataset size: {len(df_severity)} rows")
        
        # Premium prediction - full dataset
        df_premium = df.copy()
        if "CalculatedPremiumPerTerm" not in df_premium.columns:
            raise ValueError("Column 'CalculatedPremiumPerTerm' not found for premium prediction")
        print(f"  Premium dataset size: {len(df_premium)} rows")
        
        return df_severity, df[["has_claim"] + [c for c in df.columns if c != self.target]], df_premium

    def train_test_split_all(self):
        """Split all datasets into train/test sets for severity, probability, and premium models."""
        print("Performing train-test splits...")
        
        df_severity, df_claim_prob, df_premium = self.create_model_features()
        
        # Claim Severity
        print("  Severity model...")
        X_sev = df_severity.drop(columns=[self.target, "has_claim"])
        y_sev = df_severity[self.target]
        print(f"    X shape: {X_sev.shape}, y shape: {y_sev.shape}")
        X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
            X_sev, y_sev, test_size=self.test_size, random_state=self.random_state
        )
        
        # Claim Probability
        print("  Probability model...")
        # Drop columns that might not exist
        drop_cols = ["has_claim", self.target]
        existing_drop_cols = [c for c in drop_cols if c in df_claim_prob.columns]
        X_prob = df_claim_prob.drop(columns=existing_drop_cols)
        y_prob = df_claim_prob["has_claim"]
        print(f"    X shape: {X_prob.shape}, y shape: {y_prob.shape}")
        X_prob_train, X_prob_test, y_prob_train, y_prob_test = train_test_split(
            X_prob, y_prob, test_size=self.test_size, random_state=self.random_state
        )
        
        # Premium Prediction
        print("  Premium model...")
        # Drop columns that might not exist
        drop_cols = [self.target, "has_claim"]
        existing_drop_cols = [c for c in drop_cols if c in df_premium.columns]
        X_prem = df_premium.drop(columns=existing_drop_cols)
        y_prem = df_premium["CalculatedPremiumPerTerm"]
        print(f"    X shape: {X_prem.shape}, y shape: {y_prem.shape}")
        X_prem_train, X_prem_test, y_prem_train, y_prem_test = train_test_split(
            X_prem, y_prem, test_size=self.test_size, random_state=self.random_state
        )
        
        return {
            "severity": (X_sev_train, X_sev_test, y_sev_train, y_sev_test),
            "probability": (X_prob_train, X_prob_test, y_prob_train, y_prob_test),
            "premium": (X_prem_train, X_prem_test, y_prem_train, y_prem_test)
        }

    def save_processed_data(self):
        """Save train/test splits as joblib artifacts."""
        print("Saving processed data...")
        splits = self.train_test_split_all()
        
        for key, (X_train, X_test, y_train, y_test) in splits.items():
            # Convert to numpy arrays to avoid pandas issues
            X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            
            # Save the data
            joblib.dump(X_train_array, ARTIFACT_DIR / f"X_{key}_train.pkl")
            joblib.dump(X_test_array, ARTIFACT_DIR / f"X_{key}_test.pkl")
            joblib.dump(y_train_array, ARTIFACT_DIR / f"y_{key}_train.pkl")
            joblib.dump(y_test_array, ARTIFACT_DIR / f"y_{key}_test.pkl")
            
            print(f"  Saved {key}: train={X_train_array.shape}, test={X_test_array.shape}")

    def process_all(self):
        """Run full preprocessing: drop, clean, impute, and save processed data."""
        print("=" * 50)
        print("Starting data preparation...")
        print(f"Initial shape: {self.df.shape}")
        
        self.drop_extreme_missing()
        print(f"Shape after dropping columns: {self.df.shape}")
        
        self.impute_missing()
        
        self.save_processed_data()
        print("Data preparation complete. Artifacts saved in:", ARTIFACT_DIR)    