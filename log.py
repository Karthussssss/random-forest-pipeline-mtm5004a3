"""
1. Gross tonnage: Log-transform (right skewed and large values)
2. Summer deadweight: Log-transform (right skewed and large values)
3. Length: No log-transform (skewed but small values)
4. Tonnage-length ratio: No log-transform (skewed but small values)
5. Deadweight-length ratio: No log-transform (skewed but small values)
6. Deadweight-tonnage ratio: No log-transform (skewed but small values)
7. Efficiency value: No log-transform (skewed but small values)
8. Vessel age: No log-transform (skewed but small values)
9. Emission: No log-transform (outcome)

So we only do log-transform for gross tonnage and summer deadweight.
"""

import pandas as pd
import numpy as np

class LogTransformer:
    def __init__(self):
        self.features_to_log = ['GROSS_TONNAGE', 'SUMMER_DEADWEIGHT']
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to specified features."""
        df = df.copy()
        
        for feature in self.features_to_log:
            log_feature_name = f'LOG_{feature}'
            df[log_feature_name] = np.log(df[feature])
            
        return df
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse log transformation for specified features."""
        df = df.copy()
        
        for feature in self.features_to_log:
            log_feature_name = f'LOG_{feature}'
            if log_feature_name in df.columns:
                df[feature] = np.exp(df[log_feature_name])
                
        return df