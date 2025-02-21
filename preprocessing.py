import pandas as pd
import numpy as np
from typing import Tuple

class DataPreprocessor:
    def __init__(self, reference_year: int = 2024):
        self.reference_year = reference_year

    @staticmethod
    def extract_efficiency_value(x: str) -> float:
        """Extract efficiency value from string format."""
        try:
            value = x.split('(')[1].split(' ')[0]
            return float(value)
        except (IndexError, ValueError):
            return None

    def get_vessel_age(self, build_year: int) -> int:
        """Calculate vessel age from build year."""
        return self.reference_year - build_year

    def _add_efficiency_and_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency value and vessel age columns."""
        df = df.copy()
        df.loc[:, 'EFFICIENCY_VALUE'] = df['EFFICIENCY'].apply(self.extract_efficiency_value)
        df.loc[:, 'VESSEL_AGE'] = df['BUILD_YEAR'].apply(self.get_vessel_age)
        return df

    def _encode_vessel_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode vessel type."""
        df = df.copy()
        df.loc[:, 'TYPE'] = df['TYPE'].str.lower()
        type_encoded = pd.get_dummies(df['TYPE'], prefix='type', dtype=int)
        return pd.concat([df, type_encoded], axis=1)

    def _add_engineering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features based on vessel dimensions."""
        df = df.copy()
        df['DEADWEIGHT_LENGTH_RATIO'] = df['SUMMER_DEADWEIGHT'] / df['LENGTH']
        df['TONNAGE_LENGTH_RATIO'] = df['GROSS_TONNAGE'] / df['LENGTH']
        df['DEADWEIGHT_TONNAGE_RATIO'] = df['SUMMER_DEADWEIGHT'] / df['GROSS_TONNAGE']
        return df

    def preprocess(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing function that applies all transformations."""
        # Process training data
        df_train = self._add_efficiency_and_age(df_train)
        df_train = df_train.dropna(subset=['EFFICIENCY_VALUE'])  # Remove rows where we couldn't extract the value
        df_train = self._encode_vessel_type(df_train)
        df_train = self._add_engineering_features(df_train)

        # Process test data
        df_test = self._add_efficiency_and_age(df_test)
        df_test = self._encode_vessel_type(df_test)
        df_test = self._add_engineering_features(df_test)

        return df_train, df_test 