import config
from config import df_train, df_test, timestamp
from preprocessing import DataPreprocessor
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize preprocessor and preprocess data
preprocessor = DataPreprocessor()
df_train, df_test = preprocessor.preprocess(df_train.copy(), df_test.copy())

# List of features to check
features_to_check = ['GROSS_TONNAGE', 'SUMMER_DEADWEIGHT', 'LENGTH',
                     'TONNAGE_LENGTH_RATIO', 'DEADWEIGHT_LENGTH_RATIO', 
                     'DEADWEIGHT_TONNAGE_RATIO', 'EFFICIENCY_VALUE', 
                     'VESSEL_AGE', 'EMISSION']

# Create a copy of df_train for transformation checking
df_train_transform_checking = df_train.copy()

# Create a function to analyze distribution statistics
def analyze_feature_distribution(df, feature):
    stats = {
        'Mean': df[feature].mean(),
        'Median': df[feature].median(),
        'Std': df[feature].std(),
        'Skewness': df[feature].skew(),
        'Kurtosis': df[feature].kurtosis(),
        'Min': df[feature].min(),
        'Max': df[feature].max(),
        'Q1': df[feature].quantile(0.25),
        'Q3': df[feature].quantile(0.75)
    }
    return pd.Series(stats)

# Plot distributions and analyze statistics for each feature
for feature in features_to_check:
    # Distribution plots
    plt.figure(figsize=(15, 5))
    
    # Original distributions comparison
    plt.subplot(1, 3, 1)
    sns.histplot(data=df_train, x=feature, label='Train', alpha=0.5, kde=True)
    sns.histplot(data=df_test, x=feature, label='Test', alpha=0.5, kde=True)
    plt.title(f'Original {feature} Distribution\nTrain vs Test')
    plt.legend()
    
    # Apply log transformation for both datasets
    df_train_transform_checking[f'log_{feature}'] = np.log1p(df_train_transform_checking[feature])
    df_test_log = np.log1p(df_test[feature])
    
    # Plot log-transformed feature comparison
    plt.subplot(1, 3, 2)
    sns.histplot(data=df_train_transform_checking, x=f'log_{feature}', label='Train (Log)', alpha=0.5, kde=True)
    sns.histplot(data=df_test_log, label='Test (Log)', alpha=0.5, kde=True)
    plt.title(f'Log-transformed {feature}\nTrain vs Test')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(1, 3, 3)
    combined_data = pd.DataFrame({
        'value': pd.concat([df_train[feature], df_test[feature]]),
        'dataset': ['Train']*len(df_train) + ['Test']*len(df_test)
    })
    sns.boxplot(data=combined_data, x='dataset', y='value')
    plt.title(f'Boxplot of {feature}\nTrain vs Test')
    
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics for both train and test
    print(f"\nDistribution Statistics for {feature}:")
    print("=" * 50)
    print("\nTrain Set Statistics:")
    train_stats = analyze_feature_distribution(df_train, feature)
    print(train_stats)
    
    print("\nTest Set Statistics:")
    test_stats = analyze_feature_distribution(df_test, feature)
    print(test_stats)
    
    # Print extreme values
    print(f"\nTop 5 Minimum Values for {feature}:")
    print("-" * 50)
    print("Train Set:")
    print(df_train[feature].nsmallest(5))
    print("\nTest Set:")
    print(df_test[feature].nsmallest(5))
    
    print(f"\nTop 5 Maximum Values for {feature}:")
    print("-" * 50)
    print("Train Set:")
    print(df_train[feature].nlargest(5))
    print("\nTest Set:")
    print(df_test[feature].nlargest(5))
    print("\n" + "=" * 80 + "\n")
