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

# Load data
df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'train.csv'))
df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))

# Create timestamp for file naming
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
print(f'imported timestamp: {timestamp}')
