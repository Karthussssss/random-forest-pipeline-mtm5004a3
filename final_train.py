import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import joblib

class FinalTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.final_model = None
        self.running_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, 
                          best_params: dict, save_dir: str = 'results') -> None:
        """Train final model on full dataset and evaluate performance"""
        print("\n=== Training Final Model on Full Dataset ===")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Train final model
        self.final_model = RandomForestRegressor(
            **best_params, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.final_model.fit(X, y)
        
        # Make predictions
        y_pred = self.final_model.predict(X)
        
        # Calculate performance metrics
        final_r2 = r2_score(y, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print("\nFinal Model Performance on Full Dataset:")
        print(f"R² Score: {final_r2:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        
        # Save visualization
        self._save_prediction_plot(y, y_pred, save_dir)
        
        # Save model
        self._save_model(save_dir)
        
    def predict_and_save(self, X_test: pd.DataFrame, df_test: pd.DataFrame, 
                        save_dir: str = 'results') -> None:
        """Make and save predictions on test set"""
        if self.final_model is None:
            raise ValueError("Must train model before making predictions")
            
        print("\n=== Making Predictions on Test Set ===")
        test_predictions = self.final_model.predict(X_test)
        
        # Save predictions
        df_test_with_pred = df_test.copy()
        df_test_with_pred['PREDICTED_EMISSION'] = test_predictions
        predictions_file = f"{save_dir}/predictions_{self.running_timestamp}.csv"
        df_test_with_pred.to_csv(predictions_file, index=False)
        print(f"\nPredictions saved to: {predictions_file}")
        
    def _save_prediction_plot(self, y_true: pd.Series, y_pred: np.ndarray, 
                            save_dir: str) -> None:
        """Save plot of actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Emission')
        plt.xlabel('Sample Index')
        plt.ylabel('Emission')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/actual_vs_predicted_{self.running_timestamp}.pdf")
        plt.close()
        
    def _save_model(self, save_dir: str) -> None:
        """Save the trained model"""
        if self.final_model is None:
            raise ValueError("Must train model before saving it")
            
        model_file = f"{save_dir}/final_model_{self.running_timestamp}.joblib"
        joblib.dump(self.final_model, model_file)
        print(f"Final model saved to: {model_file}") 