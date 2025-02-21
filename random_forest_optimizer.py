import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from typing import Tuple, Dict, Any

class RandomForestOptimizer:
    def __init__(self, n_trials: int = 100, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_model = None
        self.study = None
        self.feature_importance = None
        
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        # Define hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = RandomForestRegressor(
                **params,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Run Optuna optimization and return best parameters"""
        def objective_wrapper(trial):
            return self.objective(trial, X_train, y_train)
        
        self.study = optuna.create_study(direction='minimize')
        print("Starting Optuna hyperparameter optimization for Random Forest...")
        self.study.optimize(objective_wrapper, n_trials=self.n_trials, show_progress_bar=True)
        
        return self.study.best_params
    
    def train_best_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the final model with best parameters"""
        if self.study is None:
            raise ValueError("Must run optimize() before training the best model")
            
        self.best_model = RandomForestRegressor(
            **self.study.best_params,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.best_model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Must train the model before making predictions")
        return self.best_model.predict(X)
    
    def plot_optimization_results(self) -> None:
        """Plot optimization results"""
        if self.study is None:
            raise ValueError("Must run optimize() before plotting results")
            
        # Optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.title('Optimization History (Random Forest)')
        plt.tight_layout()
        plt.show()
        
        # Parameter importance
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.title('Parameter Importances (Random Forest)')
        plt.tight_layout()
        plt.show()
        
        # Parallel coordinate plot
        plt.figure(figsize=(15, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(
            self.study,
            params=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
        )
        plt.title('Parallel Coordinate Plot (Random Forest)')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 10) -> None:
        """Plot feature importance"""
        if self.feature_importance is None:
            raise ValueError("Must train the model before plotting feature importance")
            
        plt.figure(figsize=(12, 6))
        plt.bar(
            self.feature_importance['feature'][:top_n],
            self.feature_importance['importance'][:top_n]
        )
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Feature Importances (Best Random Forest Model)')
        plt.tight_layout()
        plt.show()
    
    def get_model_performance(self, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, float]:
        """Calculate model performance metrics"""
        if self.best_model is None:
            raise ValueError("Must train the model before evaluating performance")
            
        y_pred = self.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        return rmse, r2 