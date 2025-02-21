import config
from config import df_train, df_test, timestamp
from preprocessing import DataPreprocessor
from outlier import OutlierRemover
from log import LogTransformer
from random_forest_optimizer import RandomForestOptimizer
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Initialize preprocessor and preprocess data
preprocessor = DataPreprocessor()
df_train, df_test = preprocessor.preprocess(df_train.copy(), df_test.copy())

# Initialize outlier remover and remove outliers first
outlier_remover = OutlierRemover()
df_train = outlier_remover.remove_outliers(df_train)

# Apply log transformation after outlier removal
log_transformer = LogTransformer()
df_train = log_transformer.transform(df_train)
df_test = log_transformer.transform(df_test)

features = ['EFFICIENCY_VALUE', 'LOG_GROSS_TONNAGE', 
           'LOG_SUMMER_DEADWEIGHT', 'LENGTH', 'VESSEL_AGE',
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
y = df_train['EMISSION'] 

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and run the Random Forest optimization
rf_optimizer = RandomForestOptimizer(n_trials=100, random_state=42)

# Optimize hyperparameters
best_params = rf_optimizer.optimize(X_train, y_train)
print("\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Train the model with best parameters
rf_optimizer.train_best_model(X_train, y_train)

# Plot optimization results
rf_optimizer.plot_optimization_results()

# Plot feature importance
rf_optimizer.plot_feature_importance(top_n=10)

# Get model performance
rmse, r2 = rf_optimizer.get_model_performance(X_val, y_val)
print("\nModel Performance on Validation Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

