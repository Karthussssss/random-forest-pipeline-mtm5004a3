import config
from config import df_train, df_test, timestamp
from preprocessing import DataPreprocessor
from outlier import OutlierRemover
from log import LogTransformer
from random_forest_optimizer import RandomForestOptimizer
from k_fold_test import KFoldTester
from final_train import FinalTrainer
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

running_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Initialize preprocessor and preprocess data
preprocessor = DataPreprocessor()
df_train, df_test = preprocessor.preprocess(df_train.copy(), df_test.copy())

# Handle missing EFFICIENCY_VALUE in test data using median from training data
efficiency_median = df_train['EFFICIENCY_VALUE'].median()
df_test.loc[:, 'EFFICIENCY_VALUE'] = df_test['EFFICIENCY_VALUE'].fillna(efficiency_median)

print("\nAfter handling missing EFFICIENCY_VALUE:")
print(f"Missing values in EFFICIENCY_VALUE: {df_test['EFFICIENCY_VALUE'].isna().sum()}")

# Initialize outlier remover and remove outliers first
outlier_remover = OutlierRemover()
df_train = outlier_remover.remove_outliers(df_train)

# Apply log transformation after outlier removal
log_transformer = LogTransformer()
df_train = log_transformer.transform(df_train)
df_test = log_transformer.transform(df_test)

features = ['EFFICIENCY_VALUE', 'GROSS_TONNAGE', 
           'SUMMER_DEADWEIGHT', 'LENGTH', 'VESSEL_AGE',
           'type_bulk carrier', 'type_chemical tanker', 'type_combination carrier',
           'type_container ship', 'type_container/ro-ro cargo ship', 'type_gas carrier',
           'type_general cargo ship', 'type_lng carrier', 'type_oil tanker',
           'type_other ship types', 'type_passenger ship', 'type_refrigerated cargo carrier',
           'type_ro-pax ship', 'type_ro-ro ship', 'type_vehicle carrier',
           'DEADWEIGHT_LENGTH_RATIO', 'TONNAGE_LENGTH_RATIO', 'DEADWEIGHT_TONNAGE_RATIO']

print("Number of features:", len(features))
print("\nFeatures to be used for model training:")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

# Split data into features and target
X = df_train[features]
y = df_train['EMISSION']  # Using original emission values as target

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and run the Random Forest optimization
print("\n=== Starting Random Forest Optimization ===")
rf_optimizer = RandomForestOptimizer(n_trials=100, random_state=42)

# Optimize hyperparameters
best_params = rf_optimizer.optimize(X_train, y_train, save_dir=results_dir)
print("\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Train the model with best parameters
rf_optimizer.train_best_model(X_train, y_train, save_dir=results_dir)

# Get model performance
rmse, r2 = rf_optimizer.get_model_performance(X_val, y_val)
print("\nInitial Model Performance on Validation Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Run K-Fold cross validation
print("\n=== Starting K-Fold Cross Validation ===")
kfold_tester = KFoldTester(n_splits=10, random_state=42)
avg_rmse, avg_r2 = kfold_tester.run_kfold_test(X, y, best_params, save_dir=results_dir)

# Ask user whether to proceed with final training
proceed = input("\nDo you want to proceed with final model training on the entire dataset? (yes/no): ")

if proceed.lower() == 'yes':
    # Initialize and run final training
    final_trainer = FinalTrainer(random_state=42)
    
    # Train and evaluate on full dataset
    final_trainer.train_and_evaluate(X, y, best_params, save_dir=results_dir)
    
    # Make predictions on test set
    print("\n=== Making Final Predictions on Test Set ===")
    X_test = df_test[features]
    y_test_pred = final_trainer.final_model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Id': range(len(y_test_pred)),
        'Predicted': y_test_pred
    })
    
    # Save predictions
    submission_file = f"{results_dir}/submission_{running_timestamp}.csv"
    submission.to_csv(submission_file, index=False)
    
    print("\nPredictions complete!")
    print(f"Submission file saved to: {submission_file}")
    print("\nSample of predictions:")
    print(submission.head())
else:
    print("\nFinal training cancelled. Exiting...")

