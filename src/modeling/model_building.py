import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from src.utility.config_loader import loader as config_loader

# Load config
config = config_loader.load("data.yaml")
ARTIFACT_DIR = Path(config["data"]["artifact_dir"])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def safe_convert_to_numeric(self, arr):
        """Safely convert array to numeric, handling all edge cases."""
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            df = pd.DataFrame(arr)
            
            # Process each column
            for col in df.columns:
                # Clean the column
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
                df[col] = df[col].replace(['', 'nan', 'NaN', 'None', 'null'], 'Unknown')
                
                # Try to convert to numeric where possible
                try:
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    # If more than 80% successfully converted, use numeric
                    if numeric_vals.notna().mean() > 0.8:
                        df[col] = numeric_vals.fillna(0)
                    else:
                        # For categorical columns, encode them
                        unique_vals = df[col].unique()
                        val_to_code = {val: i for i, val in enumerate(unique_vals)}
                        df[col] = df[col].map(val_to_code).fillna(0)
                except:
                    # Fallback: encode as categorical
                    unique_vals = df[col].unique()
                    val_to_code = {val: i for i, val in enumerate(unique_vals)}
                    df[col] = df[col].map(val_to_code).fillna(0)
            
            return df.values.astype(np.float64)
        return arr
    
    def ensure_1d(self, y):
        """Ensure y is 1D array."""
        if y.ndim > 1:
            # If it's a 2D array with multiple columns, take the first column
            print(f"  Reshaping y from {y.shape} to 1D")
            if y.shape[1] > 1:
                # For probability model, y might have multiple columns
                # Assuming first column is 'has_claim' and second is something else
                return y[:, 0].flatten()
            else:
                return y.flatten()
        return y
    
    def load_data(self):
        """Load train/test artifacts for all three models."""
        splits = {}
        
        for key in ["severity", "probability", "premium"]:
            print(f"\nLoading {key} data...")
            
            # Load the data
            X_train = joblib.load(ARTIFACT_DIR / f"X_{key}_train.pkl")
            X_test = joblib.load(ARTIFACT_DIR / f"X_{key}_test.pkl")
            y_train = joblib.load(ARTIFACT_DIR / f"y_{key}_train.pkl")
            y_test = joblib.load(ARTIFACT_DIR / f"y_{key}_test.pkl")
            
            # Convert to numeric if needed
            X_train_processed = self.safe_convert_to_numeric(X_train)
            X_test_processed = self.safe_convert_to_numeric(X_test)
            
            # Ensure y is 1D
            y_train = self.ensure_1d(y_train)
            y_test = self.ensure_1d(y_test)
            
            # For probability model, check class distribution
            if key == "probability":
                unique_classes, counts = np.unique(y_train, return_counts=True)
                print(f"  Class distribution: {dict(zip(unique_classes, counts))}")
                print(f"  Claim frequency: {counts[1]/len(y_train)*100:.2f}%" if len(counts) > 1 else "  Only one class")
            
            # Scale features
            self.scaler.fit(X_train_processed)
            X_train_scaled = self.scaler.transform(X_train_processed)
            X_test_scaled = self.scaler.transform(X_test_processed)
            
            # Ensure float64
            X_train_scaled = X_train_scaled.astype(np.float64)
            X_test_scaled = X_test_scaled.astype(np.float64)
            
            splits[key] = (X_train_scaled, X_test_scaled, y_train, y_test)
            
            print(f"  X_train shape: {X_train_scaled.shape}, dtype: {X_train_scaled.dtype}")
            print(f"  y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        
        return splits
    
    def train_severity_model(self, X_train, y_train):
        """Train regression models for claim severity."""
        print("\n" + "-"*60)
        print("Training severity models...")
        
        # Linear Regression
        print("  Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models["severity_lr"] = lr
        
        # Random Forest
        print("  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
        rf.fit(X_train, y_train)
        self.models["severity_rf"] = rf
        
        # XGBoost
        print("  Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            n_jobs=-1,
            max_depth=5,
            subsample=0.8
        )
        xgb.fit(X_train, y_train)
        self.models["severity_xgb"] = xgb
        
        print("✓ Severity models trained successfully!")
    
    def train_probability_model(self, X_train, y_train):
        """Train classification model for claim probability."""
        print("\n" + "-"*60)
        print("Training probability model...")
        
        # Check class distribution
        unique_classes = np.unique(y_train)
        print(f"  Class distribution: {np.bincount(y_train.astype(int))}")
        
        if len(unique_classes) < 2:
            print("  Warning: Only one class in training data!")
            print("  Using DummyClassifier...")
            from sklearn.dummy import DummyClassifier
            clf = DummyClassifier(strategy='constant', constant=0)
        else:
            # Use class weighting for imbalanced data
            clf = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        clf.fit(X_train, y_train)
        self.models["probability_logreg"] = clf
        
        print("✓ Probability model trained successfully!")
    
    def train_premium_model(self, X_train, y_train):
        """Train regression models for premium prediction."""
        print("\n" + "-"*60)
        print("Training premium models...")
        
        # Linear Regression
        print("  Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models["premium_lr"] = lr
        
        # Random Forest
        print("  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
        rf.fit(X_train, y_train)
        self.models["premium_rf"] = rf
        
        # XGBoost
        print("  Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            n_jobs=-1,
            max_depth=5,
            subsample=0.8
        )
        xgb.fit(X_train, y_train)
        self.models["premium_xgb"] = xgb
        
        print("✓ Premium models trained successfully!")
    
    def save_models(self):
        """Save all trained models to artifacts folder."""
        print("\n" + "-"*60)
        print("Saving models...")
        for name, model in self.models.items():
            model_path = ARTIFACT_DIR / f"{name}.pkl"
            joblib.dump(model, model_path)
            print(f"  ✓ Saved: {name}")
        
        # Save the scaler
        scaler_path = ARTIFACT_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Saved: scaler")
    
    def train_all(self):
        """Main training pipeline."""
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Load and preprocess data
            splits = self.load_data()
            
            # Train severity model
            X_sev_train, _, y_sev_train, _ = splits["severity"]
            self.train_severity_model(X_sev_train, y_sev_train)
            
            # Train probability model
            X_prob_train, _, y_prob_train, _ = splits["probability"]
            self.train_probability_model(X_prob_train, y_prob_train)
            
            # Train premium model
            X_prem_train, _, y_prem_train, _ = splits["premium"]
            self.train_premium_model(X_prem_train, y_prem_train)
            
            # Save all models
            self.save_models()
            
            print("\n" + "="*60)
            print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
            print(f"📁 Artifacts saved in: {ARTIFACT_DIR}")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()