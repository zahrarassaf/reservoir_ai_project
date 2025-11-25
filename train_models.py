"""
Main training script - RUN THIS FILE
"""
import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import generate_synthetic_spe9, build_feature_table
from src.cnn_lstm_model import build_cnn_lstm, train_cnn_lstm_model
from src.svr_model import train_svr, evaluate_svr
from src.hyperparameter_tuning import tune_svr
from src.utils import ensure_dirs

class ReservoirTrainer:
    """Main training pipeline"""
    
    def __init__(self):
        self.results = {}
        ensure_dirs()
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("📊 Loading and preparing data...")
        
        # Generate synthetic data
        df = generate_synthetic_spe9()
        features_df = build_feature_table(df)
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Time', 'Well', 'FlowRate']]
        X = features_df[feature_cols].values
        y = features_df['FlowRate'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"✅ Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_svr_model(self, X_train, y_train, X_test, y_test):
        """Train and evaluate SVR model"""
        print("📈 Training SVR model...")
        
        # Hyperparameter tuning
        best_svr, best_params = tune_svr(X_train, y_train)
        print(f"✅ Best SVR parameters: {best_params}")
        
        # Train with best parameters
        svr_trained = train_svr(X_train, y_train, 
                               C=best_params.get('C', 10.0),
                               epsilon=best_params.get('epsilon', 0.1))
        
        # Evaluate
        svr_results = evaluate_svr(svr_trained, X_test, y_test)
        
        self.results['SVR'] = {
            'model': svr_trained,
            'predictions': svr_results['y_pred'],
            'metrics': {
                'rmse': svr_results['rmse'],
                'r2': svr_results['r2']
            }
        }
        
        print(f"✅ SVR trained - RMSE: {svr_results['rmse']:.4f}, R²: {svr_results['r2']:.4f}")
        return svr_trained
    
    def train_cnn_lstm_model(self, X_train, y_train, X_test, y_test):
        """Train CNN-LSTM model (with proper sequential data)"""
        print("🧠 Training CNN-LSTM model...")
        
        # Reshape data for CNN-LSTM (samples, timesteps, features)
        # Since we don't have true sequential data, we'll create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test)
        
        if len(X_train_seq) == 0:
            print("❌ Not enough data for sequences, skipping CNN-LSTM")
            return None
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_cnn_lstm(input_shape)
        
        # Train model
        history, model_path = train_cnn_lstm_model(
            model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
            epochs=50, batch_size=16
        )
        
        # Evaluate
        y_pred = model.predict(X_test_seq).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        r2 = r2_score(y_test_seq, y_pred)
        
        self.results['CNN_LSTM'] = {
            'model': model,
            'predictions': y_pred,
            'metrics': {
                'rmse': rmse,
                'r2': r2
            }
        }
        
        print(f"✅ CNN-LSTM trained - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return model
    
    def _create_sequences(self, X, y, sequence_length=10):
        """Create sequences from tabular data"""
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i + sequence_length])
            targets.append(y[i + sequence_length])
        
        if len(sequences) == 0:
            return np.array([]), np.array([])
        
        return np.array(sequences), np.array(targets)
    
    def compare_models(self):
        """Compare model performance"""
        print("📊 Comparing model performance...")
        
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'R2': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + "="*50)
        print("MODEL COMPARISON RESULTS:")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        # Save results
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        return comparison_df
    
    def plot_results(self, X_test, y_test, feature_cols):
        """Create result visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model comparison
        plt.subplot(2, 2, 1)
        models = list(self.results.keys())
        rmse_scores = [self.results[m]['metrics']['rmse'] for m in models]
        plt.bar(models, rmse_scores, color=['skyblue', 'lightcoral'])
        plt.title('Model RMSE Comparison')
        plt.ylabel('RMSE')
        
        # Plot 2: Predictions vs Actual
        plt.subplot(2, 2, 2)
        sample_size = min(100, len(y_test))
        for model_name, result in self.results.items():
            if len(result['predictions']) >= sample_size:
                plt.plot(result['predictions'][:sample_size], 
                        label=f'{model_name} Pred', alpha=0.7)
        plt.plot(y_test[:sample_size], label='Actual', linewidth=2, color='black')
        plt.title('Predictions vs Actual')
        plt.legend()
        
        # Plot 3: Residuals
        plt.subplot(2, 2, 3)
        for model_name, result in self.results.items():
            if len(result['predictions']) == len(y_test):
                residuals = y_test - result['predictions']
                plt.hist(residuals, alpha=0.7, label=model_name, bins=20)
        plt.title('Residual Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self):
        """Execute complete training pipeline"""
        print("🚀 STARTING RESERVOIR AI TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # 1. Prepare data
            X_train, X_test, y_train, y_test, feature_cols = self.load_and_prepare_data()
            
            # 2. Train models
            self.train_svr_model(X_train, y_train, X_test, y_test)
            self.train_cnn_lstm_model(X_train, y_train, X_test, y_test)
            
            # 3. Compare and visualize
            comparison = self.compare_models()
            self.plot_results(X_test, y_test, feature_cols)
            
            # 4. Save final results
            print("💾 Results saved to 'results/' directory")
            
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            return self.results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    trainer = ReservoirTrainer()
    results = trainer.run_complete_pipeline()
