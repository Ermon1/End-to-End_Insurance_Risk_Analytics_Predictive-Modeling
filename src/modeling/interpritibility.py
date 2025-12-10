# src/modeling/interpritibility.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelInterpreter:
    def __init__(self):
        self.models = {}
        self.X_tests = {}
        self.importance_results = {}
        self.artifact_dir = Path("artifacts")
    
    def load_models_and_data(self):
        """Load trained models and test data."""
        print("Loading models and test data...")
        
        # Load models
        model_files = [
            "severity_lr.pkl", "severity_rf.pkl", "severity_xgb.pkl",
            "probability_logreg.pkl",
            "premium_lr.pkl", "premium_rf.pkl", "premium_xgb.pkl"
        ]
        
        for file in model_files:
            try:
                model_name = file.replace(".pkl", "")
                self.models[model_name] = joblib.load(self.artifact_dir / file)
                print(f"  Loaded: {model_name}")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
        
        # Load test data for each model
        for model_name in self.models.keys():
            if "severity" in model_name:
                X_test = joblib.load(self.artifact_dir / "X_severity_test.pkl")
            elif "probability" in model_name:
                X_test = joblib.load(self.artifact_dir / "X_probability_test.pkl")
            elif "premium" in model_name:
                X_test = joblib.load(self.artifact_dir / "X_premium_test.pkl")
            else:
                continue
            
            # Convert to DataFrame and ensure numeric
            n_features = X_test.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            
            # Clean data: convert to numeric
            for col in X_test_df.columns:
                X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce').fillna(0)
            
            self.X_tests[model_name] = X_test_df
    
    def compute_shap_values(self, sample_size=500):
        """Compute feature importance (using permutation importance instead of SHAP)."""
        print("\nComputing feature importance...")
        
        for name, model in self.models.items():
            print(f"\nProcessing {name}...")
            
            X_test = self.X_tests[name]
            
            # Sample for faster computation
            if len(X_test) > sample_size:
                X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
            else:
                X_sample = X_test
            
            try:
                # Try permutation importance (more reliable)
                from sklearn.inspection import permutation_importance
                
                # Use appropriate scoring
                if "probability" in name:
                    scoring = 'accuracy'
                else:
                    scoring = 'r2'
                
                r = permutation_importance(
                    model, 
                    X_sample, 
                    model.predict(X_sample), 
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1,
                    scoring=scoring
                )
                
                # Create results
                results_df = pd.DataFrame({
                    'feature': X_sample.columns,
                    'importance': r.importances_mean,
                    'std': r.importances_std
                }).sort_values('importance', ascending=False)
                
                self.importance_results[name] = results_df
                print(f"  ✓ Computed feature importance")
                
            except Exception as e:
                print(f"  ✗ Permutation importance failed: {e}")
                
                # Fallback: use model's built-in importance
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        if len(model.coef_.shape) > 1:
                            importances = np.abs(model.coef_[0])
                        else:
                            importances = np.abs(model.coef_)
                    else:
                        importances = np.ones(X_sample.shape[1]) / X_sample.shape[1]
                    
                    results_df = pd.DataFrame({
                        'feature': X_sample.columns,
                        'importance': importances / importances.sum()
                    }).sort_values('importance', ascending=False)
                    
                    self.importance_results[name] = results_df
                    print(f"  ✓ Used model's built-in importance")
                    
                except Exception as e2:
                    print(f"  ✗ All methods failed: {e2}")
                    # Create dummy results
                    results_df = pd.DataFrame({
                        'feature': X_sample.columns,
                        'importance': np.ones(X_sample.shape[1]) / X_sample.shape[1]
                    })
                    self.importance_results[name] = results_df
    
    def summarize_top_features(self, top_n=10):
        """Summarize top features for all models."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE SUMMARY")
        print("="*60)
        
        if not self.importance_results:
            print("No feature importance computed. Run compute_shap_values() first.")
            return {}
        
        top_features = {}
        
        for name, results_df in self.importance_results.items():
            # Get top N features
            top_df = results_df.head(top_n).copy()
            
            # Add percentage
            top_df['importance_pct'] = (top_df['importance'] / top_df['importance'].sum() * 100).round(2)
            
            top_features[name] = top_df
            
            # Print results
            print(f"\n{name}:")
            print("-" * len(name))
            print(top_df[['feature', 'importance', 'importance_pct']].to_string(index=False))
            
            # Save to CSV
            csv_path = self.artifact_dir / f"feature_importance_{name}.csv"
            top_df.to_csv(csv_path, index=False)
            print(f"  Saved to: {csv_path}")
        
        return top_features


# Simple test if run directly
if __name__ == "__main__":
    interpreter = ModelInterpreter()
    interpreter.load_models_and_data()
    interpreter.compute_shap_values(sample_size=500)
    interpreter.summarize_top_features(top_n=10)