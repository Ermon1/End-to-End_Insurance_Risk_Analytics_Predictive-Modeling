from src.modeling.data_load import load_task4_data
from src.modeling.data_preprocessing import DataPreparer
from src.modeling.model_building import ModelTrainer
from src.modeling.model_evalution import ModelEvaluator
from src.modeling.interpritibility import ModelInterpreter

def main():
    print("=== Task 4: End-to-End Insurance Risk Modeling ===\n")

    # # Step 1: Load Data
    print("Step 1: Loading raw data...")
    df = load_task4_data()
    print("Data loaded:", df.shape)

    # Step 2: Data Preparation
    print("\nStep 2: Preparing data (dropping, imputing, splitting)...")
    preparer = DataPreparer(df, target="TotalClaims")
    preparer.process_all()  # drops extreme missing, imputes, saves train/test splits
    print("Data preparation complete.")

    # # Step 3: Model Training
    print("\nStep 3: Training models...")
    trainer = ModelTrainer()
    trainer.train_all()
    print("Model training complete. Models saved in artifacts/")

    Step 4: Model Evaluation
    print("\nStep 4: Evaluating models...")
    evaluator = ModelEvaluator()
    evaluator.run_all_evaluations()
    evaluator.summarize_results()
    print("Model evaluation complete.")

    # Step 5: Model Interpretability
    print("\nStep 5: Feature interpretability using SHAP...")
    interpreter = ModelInterpreter()
    interpreter.load_models_and_data()
    interpreter.compute_shap_values()
    top_features = interpreter.summarize_top_features()
    
    for model_name, df_top in top_features.items():
        print(f"\nTop features for {model_name}:")
        print(df_top)

    print("\n=== Task 4 Complete ===")

if __name__ == "__main__":
    main()
