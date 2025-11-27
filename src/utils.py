"""
UTILITY FUNCTIONS FOR RESERVOIR AI PROJECT
PRODUCTION-READY HELPER FUNCTIONS
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ReservoirUtils:
    """UTILITY CLASS FOR RESERVOIR AI OPERATIONS"""
    
    @staticmethod
    def setup_plotting():
        """SETUP PROFESSIONAL PLOTTING STYLE"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    @staticmethod
    def calculate_reservoir_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """CALCULATE KEY RESERVOIR PERFORMANCE METRICS"""
        producers = df[df['well_type'] == 'PRODUCER']
        
        metrics = {
            'total_oil_production': producers['cumulative_oil'].sum(),
            'total_water_production': producers['cumulative_water'].sum(),
            'total_gas_production': producers['cumulative_gas'].sum(),
            'average_pressure': df['bottomhole_pressure'].mean(),
            'max_oil_rate': producers['oil_rate'].max(),
            'recovery_factor': producers['cumulative_oil'].sum() / (producers['cumulative_oil'].max() * len(producers) + 1e-8),
            'water_cut_trend': producers['water_rate'].mean() / (producers['oil_rate'].mean() + producers['water_rate'].mean() + 1e-8)
        }
        
        return metrics
    
    @staticmethod
    def create_performance_report(predictions: Dict, actual: np.ndarray, 
                                save_path: Path = None) -> pd.DataFrame:
        """CREATE COMPREHENSIVE PERFORMANCE REPORT"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        report_data = []
        
        for model_name, pred in predictions.items():
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            mape = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-8))) * 100
            
            report_data.append({
                'model': model_name,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mape_percent': mape,
                'std_error': np.std(actual - pred)
            })
        
        report_df = pd.DataFrame(report_data)
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"📊 PERFORMANCE REPORT SAVED: {save_path}")
        
        return report_df
    
    @staticmethod
    def plot_prediction_comparison(predictions: Dict, actual: np.ndarray, 
                                 well_id: int = None, save_path: Path = None):
        """CREATE PREDICTION COMPARISON PLOTS"""
        ReservoirUtils.setup_plotting()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # PLOT 1: ACTUAL VS PREDICTED (BEST MODEL)
        best_model = max(predictions.keys(), 
                        key=lambda x: np.corrcoef(actual, predictions[x])[0,1])
        
        axes[0].scatter(actual, predictions[best_model], alpha=0.6)
        axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Oil Rate (bbl/day)')
        axes[0].set_ylabel('Predicted Oil Rate (bbl/day)')
        axes[0].set_title(f'Actual vs Predicted - {best_model}')
        
        # PLOT 2: TIME SERIES COMPARISON
        time_points = min(200, len(actual))
        for model_name, pred in list(predictions.items())[:3]:  # Top 3 models
            axes[1].plot(range(time_points), pred[:time_points], 
                        label=model_name, alpha=0.7)
        
        axes[1].plot(range(time_points), actual[:time_points], 
                    'k-', label='Actual', linewidth=2)
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Oil Rate (bbl/day)')
        axes[1].set_title('Time Series Predictions')
        axes[1].legend()
        
        # PLOT 3: ERROR DISTRIBUTION
        errors = {}
        for model_name, pred in predictions.items():
            errors[model_name] = actual - pred
        
        axes[2].boxplot(errors.values(), labels=errors.keys())
        axes[2].set_ylabel('Prediction Error (bbl/day)')
        axes[2].set_title('Error Distribution by Model')
        axes[2].tick_params(axis='x', rotation=45)
        
        # PLOT 4: PERFORMANCE COMPARISON
        performance = []
        for model_name, pred in predictions.items():
            r2 = r2_score(actual, pred)
            performance.append((model_name, r2))
        
        models, scores = zip(*sorted(performance, key=lambda x: x[1], reverse=True))
        axes[3].barh(range(len(models)), scores)
        axes[3].set_yticks(range(len(models)))
        axes[3].set_yticklabels(models)
        axes[3].set_xlabel('R² Score')
        axes[3].set_title('Model Performance Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 PLOT SAVED: {save_path}")
        
        plt.show()
    
    @staticmethod
    def save_training_artifacts(history: Dict, model, feature_names: List[str],
                              save_dir: Path = config.RESULTS_DIR):
        """SAVE TRAINING ARTIFACTS FOR REPRODUCIBILITY"""
        artifacts_dir = save_dir / 'training_artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        
        # SAVE TRAINING HISTORY
        history_df = pd.DataFrame(history)
        history_df.to_csv(artifacts_dir / 'training_history.csv', index=False)
        
        # SAVE FEATURE INFORMATION
        feature_info = {
            'feature_names': feature_names,
            'num_features': len(feature_names),
            'feature_types': 'mixed'
        }
        
        with open(artifacts_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # SAVE MODEL SUMMARY
        if hasattr(model, 'summary'):
            with open(artifacts_dir / 'model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"📚 TRAINING ARTIFACTS SAVED: {artifacts_dir}")

def setup_logging():
    """SETUP PROFESSIONAL LOGGING"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reservoir_ai.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
