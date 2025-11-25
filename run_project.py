"""
MAIN EXECUTION FILE - Run this to train models
FIXED VERSION - Correct imports and error handling
"""
import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Fix import path - Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.data_preprocessing import generate_synthetic_spe9, build_feature_table
    from src.cnn_lstm_model import build_cnn_lstm, train_cnn_lstm_model
    from src.svr_model import train_svr, evaluate_svr
    from src.hyperparameter_tuning import tune_svr
    from src.utils import ensure_dirs
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import...")
    # Alternative import attempt
    from data_preprocessing import generate_synthetic_spe9, build_feature_table
    from cnn_lstm_model import build_cnn_lstm, train_cnn_lstm_model
    from svr_model import train_svr, evaluate_svr
    from hyperparameter_tuning import tune_svr
    from utils import ensure_dirs

def create_sequences(X, y, sequence_length=10):
    """Create sequences from tabular data for CNN-LSTM"""
    sequences = []
    targets = []
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i + sequence_length])
        targets.append(y[i + sequence_length])
    
    if len(sequences) == 0:
        return np.array([]), np.array([])
    
    return np.array(sequences), np.array(targets)

def main():
    """Main training pipeline"""
    print("🚀 Starting Reservoir AI Project - FIXED VERSION")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_dirs()
    
    try:
        # 1. Generate and load data
        print("📊 Step 1: Generating synthetic SPE9 data...")
        df = generate_synthetic_spe9()
        features_df = build_feature_table(df)
        
        print(f"✅ Data generated: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        
        # 2. Prepare features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Time', 'Well', 'FlowRate']]
        X = features_df[feature_cols].values
        y = features_df['FlowRate'].values
        
        print(f"Features: {len(feature_cols)}, Target shape: {y.shape}")
        
        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"✅ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 4. Train SVR model
        print("\n📈 Step 2: Training SVR model...")
        best_svr, best_params = tune_svr(X_train, y_train)
        print(f"✅ Best SVR parameters: {best_params}")
        
        svr_trained = train_svr(X_train, y_train, 
                               C=best_params.get('C', 10.0),
                               epsilon=best_params.get('epsilon', 0.1))
        
        svr_results = evaluate_svr(svr_trained, X_test, y_test)
        print(f"✅ SVR Results - RMSE: {svr_results['rmse']:.4f}, R²: {svr_results['r2']:.4f}")
        
        # 5. Train CNN-LSTM (with sequential data)
        print("\n🧠 Step 3: Training CNN-LSTM model...")
        
        # Create sequential data
        X_train_seq, y_train_seq = create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test)
        
        print(f"Sequential data - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
        
        if len(X_train_seq) > 0:
            # Build and train CNN-LSTM
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            print(f"Building CNN-LSTM with input shape: {input_shape}")
            
            cnn_lstm_model = build_cnn_lstm(input_shape)
            
            history, model_path = train_cnn_lstm_model(
                cnn_lstm_model, X_train_seq, y_train_seq, 
                X_test_seq, y_test_seq, epochs=50, batch_size=16
            )
            
            # Evaluate CNN-LSTM
            y_pred_cnn = cnn_lstm_model.predict(X_test_seq).flatten()
            cnn_rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred_cnn))
            cnn_r2 = r2_score(y_test_seq, y_pred_cnn)
            print(f"✅ CNN-LSTM Results - RMSE: {cnn_rmse:.4f}, R²: {cnn_r2:.4f}")
        else:
            print("⚠️ Not enough data for CNN-LSTM sequences")
            y_pred_cnn = None
            cnn_rmse = cnn_r2 = np.nan
        
        # 6. Compare results
        print("\n📊 Step 4: Model Comparison")
        print("=" * 40)
        results_data = [
            {"Model": "SVR", "RMSE": svr_results['rmse'], "R²": svr_results['r2']},
            {"Model": "CNN-LSTM", "RMSE": cnn_rmse, "R²": cnn_r2}
        ]
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # 7. Save results
        results_df.to_csv('results/model_results.csv', index=False)
        joblib.dump(svr_trained['model'], 'models/svr_model.pkl')
        
        # 8. Create visualizations
        print("\n📈 Step 5: Creating visualizations...")
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model comparison
        plt.subplot(2, 3, 1)
        models = ['SVR', 'CNN-LSTM']
        rmse_values = [svr_results['rmse'], cnn_rmse if not np.isnan(cnn_rmse) else 0]
        colors = ['skyblue', 'lightcoral']
        plt.bar(models, rmse_values, color=colors)
        plt.title('Model RMSE Comparison')
        plt.ylabel('RMSE')
        
        # Plot 2: R² comparison
        plt.subplot(2, 3, 2)
        r2_values = [svr_results['r2'], cnn_r2 if not np.isnan(cnn_r2) else 0]
        plt.bar(models, r2_values, color=colors)
        plt.title('R² Score Comparison')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        # Plot 3: Predictions vs Actual
        plt.subplot(2, 3, 3)
        sample_size = min(50, len(y_test))
        plt.plot(y_test[:sample_size], label='Actual', linewidth=2, color='black')
        plt.plot(svr_results['y_pred'][:sample_size], label='SVR Pred', alpha=0.7)
        if y_pred_cnn is not None and len(y_pred_cnn) >= sample_size:
            plt.plot(y_pred_cnn[:sample_size], label='CNN-LSTM Pred', alpha=0.7)
        plt.title('Predictions vs Actual')
        plt.legend()
        
        # Plot 4: Residuals SVR
        plt.subplot(2, 3, 4)
        svr_residuals = y_test - svr_results['y_pred']
        plt.hist(svr_residuals, bins=20, alpha=0.7, color='skyblue')
        plt.title('SVR Residual Distribution')
        plt.xlabel('Residuals')
        
        # Plot 5: Feature importance
        plt.subplot(2, 3, 5)
        if hasattr(svr_trained['model'], 'coef_'):
            importance = np.abs(svr_trained['model'].coef_)
            if len(importance) > 0:
                top_indices = np.argsort(importance)[-8:]  # Top 8 features
                top_features = [feature_cols[i] for i in top_indices]
                plt.barh(range(len(top_features)), importance[top_indices])
                plt.yticks(range(len(top_features)), top_features)
                plt.title('Top Feature Importance (SVR)')
        
        # Plot 6: Data distribution
        plt.subplot(2, 3, 6)
        plt.hist(y, bins=30, alpha=0.7, color='lightgreen')
        plt.title('Target Variable Distribution')
        plt.xlabel('FlowRate')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n🎉 PROJECT EXECUTED SUCCESSFULLY!")
        print("📁 Results saved to:")
        print("   - results/model_results.csv")
        print("   - results/model_comparison.png") 
        print("   - models/svr_model.pkl")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
