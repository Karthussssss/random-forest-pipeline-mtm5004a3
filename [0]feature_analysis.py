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
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_train[feature], kde=True)
    plt.title(f'Original {feature} (After Outlier Removal)')
    
    # Apply log transformation
    df_train_transform_checking[f'log_{feature}'] = np.log1p(df_train_transform_checking[feature])
    
    # Plot log-transformed feature
    plt.subplot(1, 2, 2)
    sns.histplot(df_train_transform_checking[f'log_{feature}'], kde=True)
    plt.title(f'Log-transformed {feature} (After Outlier Removal)')
    
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics
    print(f"\nDistribution Statistics for {feature}:")
    print("=" * 50)
    stats = analyze_feature_distribution(df_train, feature)
    print(stats)
    
    # Print extreme values
    print(f"\nTop 10 Minimum Values for {feature}:")
    print("-" * 50)
    print(df_train[feature].nsmallest(10))
    
    print(f"\nTop 10 Maximum Values for {feature}:")
    print("-" * 50)
    print(df_train[feature].nlargest(10))
    print("\n" + "=" * 80 + "\n")

# Create box plots for original features
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
df_train_melted = df_train[features_to_check].melt()
sns.boxplot(x='variable', y='value', data=df_train_melted)
plt.xticks(rotation=45)
plt.title('Box Plots of Original Features')
plt.tight_layout()

# Create box plots for log-transformed features
plt.subplot(1, 2, 2)
df_log_melted = df_train_transform_checking[[f'log_{feature}' for feature in features_to_check]].melt()
sns.boxplot(x='variable', y='value', data=df_log_melted)
plt.xticks(rotation=45)
plt.title('Box Plots of Log-transformed Features')
plt.tight_layout()
plt.show()

# Create individual box plots for better detail
for feature in features_to_check:
    plt.figure(figsize=(12, 5))
    
    # Original feature box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_train[feature])
    plt.title(f'Box Plot of {feature}')
    
    # Log-transformed feature box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_train_transform_checking[f'log_{feature}'])
    plt.title(f'Box Plot of Log-transformed {feature}')
    
    plt.tight_layout()
    plt.show()

# Calculate and print outlier information
def calculate_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        'Total Outliers': len(outliers),
        'Percentage Outliers': (len(outliers) / len(series)) * 100,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    }

print("\nOutlier Analysis")
print("=" * 80)
for feature in features_to_check:
    print(f"\nOutlier Information for {feature}:")
    print("-" * 50)
    outlier_stats = calculate_outliers(df_train[feature])
    print(f"Number of outliers: {outlier_stats['Total Outliers']}")
    print(f"Percentage of outliers: {outlier_stats['Percentage Outliers']:.2f}%")
    print(f"Lower bound: {outlier_stats['Lower Bound']:.2f}")
    print(f"Upper bound: {outlier_stats['Upper Bound']:.2f}")


