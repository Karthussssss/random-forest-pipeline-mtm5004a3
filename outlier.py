import pandas as pd
import numpy as np

class OutlierRemover:
    """
    A class to handle outlier removal for vessel data preprocessing.
    
    Outlier removal rules:
    1. EMISSION: between 1 and 4400
    2. LENGTH: <= 500 (to match test set range)
    3. GROSS_TONNAGE: no specific limits (no significant outliers)
    4. EFFICIENCY_VALUE: <= 1000
    5. SUMMER_DEADWEIGHT: <= 330,000
    6. TONNAGE_LENGTH_RATIO: <= 3000
    7. DEADWEIGHT_LENGTH_RATIO: <= 6000
    """
    
    def __init__(self):
        self.rules = {
            'EMISSION': {'min': 1, 'max': 4400},
            'LENGTH': {'max': 500},
            'EFFICIENCY_VALUE': {'max': 1000},
            'SUMMER_DEADWEIGHT': {'max': 330000},
            'TONNAGE_LENGTH_RATIO': {'max': 3000},
            'DEADWEIGHT_LENGTH_RATIO': {'max': 6000}
        }
        self.initial_rows = 0
        self.final_rows = 0
    
    def remove_outliers(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Remove outliers from the dataset based on predefined rules.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            verbose (bool): Whether to print removal statistics
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        self.initial_rows = len(df)
        df_clean = df.copy()
        
        # Apply each rule
        for feature, limits in self.rules.items():
            original_len = len(df_clean)
            
            # Apply minimum limit if it exists
            if 'min' in limits:
                df_clean = df_clean[df_clean[feature] >= limits['min']]
            
            # Apply maximum limit if it exists
            if 'max' in limits:
                df_clean = df_clean[df_clean[feature] <= limits['max']]
            
            if verbose:
                rows_removed = original_len - len(df_clean)
                if rows_removed > 0:
                    print(f"Removed {rows_removed} rows ({(rows_removed/original_len)*100:.2f}% of data) "
                          f"due to {feature} outliers")
        
        self.final_rows = len(df_clean)
        
        if verbose:
            total_removed = self.initial_rows - self.final_rows
            print(f"\nTotal rows removed: {total_removed} "
                  f"({(total_removed/self.initial_rows)*100:.2f}% of original data)")
        
        return df_clean
    
    def get_removal_stats(self) -> dict:
        """
        Get statistics about the number of rows removed.
        
        Returns:
            dict: Dictionary containing removal statistics
        """
        return {
            'initial_rows': self.initial_rows,
            'final_rows': self.final_rows,
            'removed_rows': self.initial_rows - self.final_rows,
            'removal_percentage': ((self.initial_rows - self.final_rows) / self.initial_rows) * 100
        } 