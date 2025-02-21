import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple
import os
from datetime import datetime

class KFoldTester:
    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.rmse_scores = []
        self.r2_scores = []
        self.final_model = None
        
    def run_kfold_test(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       best_params: Dict[str, Any], save_dir: str = 'results') -> Tuple[float, float]:
        """Run k-fold cross validation and return average metrics"""
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        print(f"\nStarting {self.n_splits}-fold cross-validation...")
        
        # Perform K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            # Split data for this fold
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Initialize and train model with best parameters
            rf_model = RandomForestRegressor(**best_params, random_state=self.random_state, n_jobs=-1)
            rf_model.fit(X_fold_train, y_fold_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_fold_val)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            r2 = r2_score(y_fold_val, y_pred)
            
            # Store scores
            self.rmse_scores.append(rmse)
            self.r2_scores.append(r2)
            
            print(f"\nFold {fold}:")
            print(f"RMSE: {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
        
        # Calculate and print average performance
        avg_rmse = np.mean(self.rmse_scores)
        avg_r2 = np.mean(self.r2_scores)
        print("\nOverall Cross-validation Performance:")
        print(f"Average RMSE: {avg_rmse:.2f} (±{np.std(self.rmse_scores):.2f})")
        print(f"Average R² Score: {avg_r2:.4f} (±{np.std(self.r2_scores):.4f})")
        
        # Train final model on full training data
        self.final_model = RandomForestRegressor(**best_params, random_state=self.random_state, n_jobs=-1)
        self.final_model.fit(X_train, y_train)
        
        # Save plots
        self._save_cv_plots(save_dir, timestamp)
        self._save_feature_importance_plot(X_train.columns, save_dir, timestamp)
        
        return avg_rmse, avg_r2
    
    def _save_cv_plots(self, save_dir: str, timestamp: str) -> None:
        """Save cross-validation results plots"""
        plt.figure(figsize=(12, 5))
        
        # Plot RMSE scores
        plt.subplot(1, 2, 1)
        plt.boxplot(self.rmse_scores)
        plt.title('RMSE Scores Distribution')
        plt.ylabel('RMSE')
        
        # Plot R² scores
        plt.subplot(1, 2, 2)
        plt.boxplot(self.r2_scores)
        plt.title('R² Scores Distribution')
        plt.ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cv_results_{timestamp}.pdf")
        plt.close()
    
    def _save_feature_importance_plot(self, feature_names: pd.Index, save_dir: str, timestamp: str) -> None:
        """Save feature importance plot"""
        if self.final_model is None:
            raise ValueError("Must run k-fold test before plotting feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Feature Importances (Final Model)')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance_{timestamp}.pdf")
        plt.close()
    
    def get_final_model(self) -> RandomForestRegressor:
        """Return the final model trained on full training data"""
        if self.final_model is None:
            raise ValueError("Must run k-fold test before getting final model")
        return self.final_model 