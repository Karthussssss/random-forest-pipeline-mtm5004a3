import config
from config import df_train, df_test, timestamp
from preprocessing import DataPreprocessor
from outlier import OutlierRemover
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

# Initialize outlier remover and remove outliers
outlier_remover = OutlierRemover()
df_train = outlier_remover.remove_outliers(df_train)

