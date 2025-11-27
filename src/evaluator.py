"""
PRODUCTION MODEL EVALUATION SYSTEM
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ModelEvaluator:
    """COMPREHENSIVE MODEL EVALUATION"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_predictions(self, predictions: dict, actual: np.ndarray) -> pd.DataFrame:
        """EVALUATE ALL MODEL PREDICTIONS"""
        evaluation_data = []
        
        for model_name, y_pred in predictions.items():
            if len(y_pred) != len(actual):
                continue
                
            mae = mean_absolute_error(actual, y_pred)
            rmse = np.sqrt(mean_squared_error(actual, y_pred))
            r2 = r2_score(actual, y_pred)
            mape = np.mean(np.abs((actual - y_pred) / (np.abs(actual) + 1e-8))) * 100
            
            evaluation_data.append({
                'model': model_name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            })
        
        results_df = pd.DataFrame(evaluation_data)
        self.results = results_df.to_dict('records')
        
        return results_df
    
    def save_evaluation_results(self, results_df: pd.DataFrame, filename: str = "model_performance.csv"):
        """SAVE EVALUATION RESULTS"""
        results_path = config.RESULTS_DIR / filename
        results_df.to_csv(results_path, index=False)
        print(f"💾 EVALUATION RESULTS SAVED: {results_path}")
    
    def print_performance_summary(self, results_df: pd.DataFrame):
        """PRINT PERFORMANCE SUMMARY"""
        print("\n🎯 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'MODEL':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
        print("-" * 60)
        
        for _, row in results_df.iterrows():
            print(f"{row['model']:<20} {row['mae']:<10.1f} {row['rmse']:<10.1f} "
                  f"{row['r2']:<10.3f} {row['mape']:<10.1f}%")
        
        best_model = results_df.loc[results_df['r2'].idxmax()]
        print("=" * 60)
        print(f"🏆 BEST MODEL: {best_model['model']}")
        print(f"   R² Score: {best_model['r2']:.3f}")
        print(f"   MAE: {best_model['mae']:.1f} bbl/day")
