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

# Apply log transformation to all features
for feature in features_to_check:
    df_train_transform_checking[f'log_{feature}'] = np.log1p(df_train_transform_checking[feature])

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

# Function to remove outliers based on IQR method
def remove_outliers(df, features):
    df_clean = df.copy()
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
    return df_clean

# Create clean dataset with outliers removed
df_train_clean = remove_outliers(df_train, features_to_check)

# Plot distributions and analyze statistics for each feature
for feature in features_to_check:
    # Distribution plots (3 subplots: original, log-transformed, and cleaned)
    plt.figure(figsize=(15, 5))
    
    # Original distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df_train[feature], kde=True)
    plt.title(f'Original {feature}\n(Before Outlier Removal)')
    
    # Log-transformed distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df_train_transform_checking[f'log_{feature}'], kde=True)
    plt.title(f'Log-transformed {feature}\n(Before Outlier Removal)')
    
    # Cleaned distribution
    plt.subplot(1, 3, 3)
    sns.histplot(df_train_clean[feature], kde=True)
    plt.title(f'Original {feature}\n(After Outlier Removal)')
    
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics for both original and cleaned data
    print(f"\nDistribution Statistics for {feature}:")
    print("=" * 50)
    print("\nBefore Outlier Removal:")
    print("-" * 30)
    stats_before = analyze_feature_distribution(df_train, feature)
    print(stats_before)
    
    print("\nAfter Outlier Removal:")
    print("-" * 30)
    stats_after = analyze_feature_distribution(df_train_clean, feature)
    print(stats_after)
    
    # Print extreme values for both datasets
    print(f"\nTop 10 Minimum Values for {feature} (Before Removal):")
    print("-" * 50)
    print(df_train[feature].nsmallest(10))
    
    print(f"\nTop 10 Maximum Values for {feature} (Before Removal):")
    print("-" * 50)
    print(df_train[feature].nlargest(10))
    
    print(f"\nTop 10 Minimum Values for {feature} (After Removal):")
    print("-" * 50)
    print(df_train_clean[feature].nsmallest(10))
    
    print(f"\nTop 10 Maximum Values for {feature} (After Removal):")
    print("-" * 50)
    print(df_train_clean[feature].nlargest(10))
    print("\n" + "=" * 80 + "\n")

# Create box plots comparing before and after outlier removal
plt.figure(figsize=(15, 8))

# Before removal
plt.subplot(1, 2, 1)
df_train_melted = df_train[features_to_check].melt()
sns.boxplot(x='variable', y='value', data=df_train_melted)
plt.xticks(rotation=45)
plt.title('Box Plots Before Outlier Removal')

# After removal
plt.subplot(1, 2, 2)
df_train_clean_melted = df_train_clean[features_to_check].melt()
sns.boxplot(x='variable', y='value', data=df_train_clean_melted)
plt.xticks(rotation=45)
plt.title('Box Plots After Outlier Removal')

plt.tight_layout()
plt.show()

# Create individual box plots comparing before and after outlier removal
for feature in features_to_check:
    plt.figure(figsize=(15, 5))
    
    # Original feature box plot
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df_train[feature])
    plt.title(f'Box Plot of {feature}\n(Before Removal)')
    
    # Log-transformed feature box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df_train_transform_checking[f'log_{feature}'])
    plt.title(f'Box Plot of Log-transformed\n{feature}')
    
    # Cleaned feature box plot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df_train_clean[feature])
    plt.title(f'Box Plot of {feature}\n(After Removal)')
    
    plt.tight_layout()
    plt.show()

# Print summary statistics about data reduction
print("\nData Reduction Summary")
print("=" * 80)
print(f"Original dataset size: {len(df_train)}")
print(f"Cleaned dataset size: {len(df_train_clean)}")
print(f"Removed records: {len(df_train) - len(df_train_clean)}")
print(f"Percentage of data retained: {(len(df_train_clean) / len(df_train)) * 100:.2f}%")


