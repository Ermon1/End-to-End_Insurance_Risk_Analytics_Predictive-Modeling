# src/modeling/model_evaluation.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, classification_report

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.splits = {}
        self.results = {}
        self.artifact_dir = Path("artifacts")
    
    def clean_numeric_data(self, X):
        """Clean numeric data by converting all values to float."""
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier cleaning
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            if len(X.shape) > 1:
                X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
            else:
                X_df = pd.DataFrame(X, columns=["target"])
        else:
            X_df = X.copy()
        
        # Convert all columns to numeric
        for col in X_df.columns:
            # First convert to string, clean, then to numeric
            X_df[col] = X_df[col].astype(str).str.strip()
            X_df[col] = X_df[col].replace(['', 'nan', 'NaN', 'None', 'null', '  ', '   '], '0')
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        # Convert back to numpy array if input was numpy array
        if isinstance(X, np.ndarray):
            return X_df.values.astype(np.float64)
        return X_df
    
    def load_models(self):
        """Load all models from artifacts."""
        print("Loading trained models...")
        
        model_files = [
            "severity_lr.pkl", "severity_rf.pkl", "severity_xgb.pkl",
            "probability_logreg.pkl",
            "premium_lr.pkl", "premium_rf.pkl", "premium_xgb.pkl"
        ]
        
        for file in model_files:
            try:
                model_name = file.replace(".pkl", "")
                self.models[model_name] = joblib.load(self.artifact_dir / file)
                print(f"  ✓ Loaded: {model_name}")
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")
    
    def load_data(self):
        """Load and clean test data for evaluation."""
        print("\nLoading and cleaning test data...")
        
        for key in ["severity", "probability", "premium"]:
            try:
                X_test = joblib.load(self.artifact_dir / f"X_{key}_test.pkl")
                y_test = joblib.load(self.artifact_dir / f"y_{key}_test.pkl")
                
                # Clean the data
                print(f"  Cleaning {key} data...")
                X_test_clean = self.clean_numeric_data(X_test)
                y_test_clean = self.clean_numeric_data(y_test).flatten()
                
                # Check for string values
                if isinstance(X_test, np.ndarray) and X_test.dtype == object:
                    print(f"    Warning: {key} X_test had object dtype, converted to float64")
                
                self.splits[key] = (X_test_clean, y_test_clean)
                print(f"  ✓ Loaded {key}: X_test={X_test_clean.shape}, y_test={y_test_clean.shape}")
                
            except Exception as e:
                print(f"  ✗ Error loading {key} data: {e}")
    
    def evaluate_regression(self, model_name, X_test, y_test):
        """Evaluate regression models."""
        try:
            model = self.models[model_name]
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"    RMSE: {rmse:,.2f}")
            print(f"    R²: {r2:.4f}")
            print(f"    Predictions range: {y_pred.min():,.2f} - {y_pred.max():,.2f}")
            print(f"    Actual range: {y_test.min():,.2f} - {y_test.max():,.2f}")
            
            return {
                "RMSE": rmse,
                "R2": r2,
                "y_pred_min": y_pred.min(),
                "y_pred_max": y_pred.max(),
                "y_test_min": y_test.min(),
                "y_test_max": y_test.max()
            }
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {"RMSE": None, "R2": None}
    
    def evaluate_classification(self, model_name, X_test, y_test):
        """Evaluate classification model."""
        try:
            model = self.models[model_name]
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"    Accuracy: {accuracy:.4f}")
            
            # Calculate class distribution
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            unique_true, counts_true = np.unique(y_test, return_counts=True)
            
            print(f"    Predictions: {dict(zip(unique_pred, counts_pred))}")
            print(f"    Actual: {dict(zip(unique_true, counts_true))}")
            
            # Try to get ROC-AUC if possible
            roc_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    print(f"    ROC-AUC: {roc_auc:.4f}")
                except:
                    print("    ROC-AUC: Could not compute (likely only one class)")
            
            # Print classification report
            print("\n    Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0, target_names=['No Claim', 'Claim']))
            
            return {
                "Accuracy": accuracy,
                "ROC-AUC": roc_auc,
                "Predicted_0": counts_pred[0] if 0 in unique_pred else 0,
                "Predicted_1": counts_pred[1] if 1 in unique_pred else 0,
                "Actual_0": counts_true[0] if 0 in unique_true else 0,
                "Actual_1": counts_true[1] if 1 in unique_true else 0
            }
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return {"Accuracy": None, "ROC-AUC": None}
    
    def run_all_evaluations(self):
        """Run evaluations for all models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        self.load_models()
        self.load_data()
        
        # Evaluate severity models
        print("\n" + "-"*60)
        print("SEVERITY MODELS (Claim Amount Prediction)")
        print("-"*60)
        
        if "severity" in self.splits:
            X_test, y_test = self.splits["severity"]
            for model_name in ["severity_lr", "severity_rf", "severity_xgb"]:
                if model_name in self.models:
                    print(f"\n{model_name}:")
                    result = self.evaluate_regression(model_name, X_test, y_test)
                    self.results[model_name] = result
        
        # Evaluate probability model
        print("\n" + "-"*60)
        print("PROBABILITY MODEL (Claim Occurrence Prediction)")
        print("-"*60)
        
        if "probability" in self.splits:
            X_test, y_test = self.splits["probability"]
            model_name = "probability_logreg"
            if model_name in self.models:
                print(f"\n{model_name}:")
                result = self.evaluate_classification(model_name, X_test, y_test)
                self.results[model_name] = result
        
        # Evaluate premium models
        print("\n" + "-"*60)
        print("PREMIUM MODELS (Premium Amount Prediction)")
        print("-"*60)
        
        if "premium" in self.splits:
            X_test, y_test = self.splits["premium"]
            for model_name in ["premium_lr", "premium_rf", "premium_xgb"]:
                if model_name in self.models:
                    print(f"\n{model_name}:")
                    result = self.evaluate_regression(model_name, X_test, y_test)
                    self.results[model_name] = result
    
    def summarize_results(self):
        """Summarize and display evaluation results."""
        if not self.results:
            print("\nNo evaluation results available.")
            return None
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Create summary DataFrame
        summary_data = {}
        for model_name, metrics in self.results.items():
            # Extract key metrics
            model_summary = {}
            
            if "severity" in model_name or "premium" in model_name:
                # Regression models
                model_summary["RMSE"] = metrics.get("RMSE")
                model_summary["R2"] = metrics.get("R2")
                model_summary["Pred_Min"] = metrics.get("y_pred_min")
                model_summary["Pred_Max"] = metrics.get("y_pred_max")
            
            elif "probability" in model_name:
                # Classification model
                model_summary["Accuracy"] = metrics.get("Accuracy")
                model_summary["ROC-AUC"] = metrics.get("ROC-AUC")
                model_summary["Pred_Claims"] = metrics.get("Predicted_1")
                model_summary["Actual_Claims"] = metrics.get("Actual_1")
            
            summary_data[model_name] = model_summary
        
        # Create and display DataFrame
        df_summary = pd.DataFrame(summary_data).T
        
        # Format for nice display
        pd.set_option('display.float_format', '{:,.4f}'.format)
        pd.set_option('display.max_columns', None)
        
        print("\nSummary of Model Performance:")
        print(df_summary)
        
        # Save to CSV
        csv_path = self.artifact_dir / "model_evaluation_summary.csv"
        df_summary.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Generate insights
        self.generate_insights()
        
        return df_summary
    
    def generate_insights(self):
        """Generate business insights from evaluation results."""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS")
        print("="*60)
        
        # Severity model insights
        severity_models = [m for m in self.results.keys() if "severity" in m]
        if severity_models:
            valid_models = [(m, self.results[m]["RMSE"]) for m in severity_models 
                          if self.results[m]["RMSE"] is not None]
            if valid_models:
                best_severity = min(valid_models, key=lambda x: x[1])
                print(f"\n• Claim Severity Prediction:")
                print(f"  Best model: {best_severity[0]} (RMSE: {best_severity[1]:,.2f})")
                print(f"  Interpretation: Predicts claim amounts within ±{best_severity[1]:,.0f}")
        
        # Probability model insights
        if "probability_logreg" in self.results:
            prob_result = self.results["probability_logreg"]
            if prob_result.get("Accuracy") is not None:
                accuracy = prob_result["Accuracy"]
                pred_claims = prob_result.get("Predicted_1", 0)
                actual_claims = prob_result.get("Actual_1", 0)
                
                print(f"\n• Claim Probability Prediction:")
                print(f"  Accuracy: {accuracy:.2%}")
                print(f"  Predicted {pred_claims:,} claims (actual: {actual_claims:,})")
                
                if pred_claims > 0:
                    detection_rate = pred_claims / actual_claims if actual_claims > 0 else 0
                    print(f"  Claim detection rate: {detection_rate:.2%}")
        
        # Premium model insights
        premium_models = [m for m in self.results.keys() if "premium" in m]
        if premium_models:
            valid_models = [(m, self.results[m]["R2"]) for m in premium_models 
                          if self.results[m]["R2"] is not None]
            if valid_models:
                best_premium = max(valid_models, key=lambda x: x[1])
                r2 = best_premium[1]
                print(f"\n• Premium Prediction:")
                print(f"  Best model: {best_premium[0]} (R²: {r2:.4f})")
                
                if r2 > 0.7:
                    print(f"  Excellent predictive power")
                elif r2 > 0.5:
                    print(f"  Good predictive power")
                elif r2 > 0.3:
                    print(f"  Moderate predictive power")
                else:
                    print(f"  Limited predictive power")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # General recommendations
        print("\n1. Data Quality:")
        print("   • Ensure all data is properly cleaned before modeling")
        print("   • Handle missing values and string data appropriately")
        
        print("\n2. Model Selection:")
        print("   • Use the best model from each category for production")
        print("   • Consider ensemble methods for improved accuracy")
        
        print("\n3. Business Applications:")
        print("   • Use severity model for reserve setting")
        print("   • Use probability model for risk assessment")
        print("   • Use premium model for pricing optimization")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_all_evaluations()
    evaluator.summarize_results()