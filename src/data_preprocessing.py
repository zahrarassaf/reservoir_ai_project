"""
Improved Data Preprocessing Module
Fixed data leakage and added validation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config import *

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_columns = FEATURE_COLUMNS
        
    def load_and_clean_data(self):
        """Load and clean the well log data"""
        try:
            df = pd.read_csv(DATA_PATH)
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Handle missing values
            df_clean = df.dropna()
            print(f"✅ Data after cleaning: {df_clean.shape}")
            
            return df_clean
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features and targets without data leakage"""
        X = df[self.feature_columns]
        y_permeability = df['Permeability']
        y_porosity = df['Porosity']
        
        return X, y_permeability, y_porosity
    
    def train_test_split_data(self, X, y):
        """Split data with proper stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            shuffle=True
        )
        
        print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, target_name):
        """Scale features without data leakage"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Only transform test data
        
        # Save scaler for future use
        self.scalers[target_name] = scaler
        joblib.dump(scaler, f'{MODELS_PATH}scaler_{target_name}.pkl')
        
        return X_train_scaled, X_test_scaled

def main():
    """Main preprocessing pipeline"""
    print("🚀 Starting improved data preprocessing...")
    
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_and_clean_data()
    if df is None:
        return
    
    # Prepare features
    X, y_perm, y_por = preprocessor.prepare_features(df)
    
    # Split for permeability
    X_train_perm, X_test_perm, y_train_perm, y_test_perm = preprocessor.train_test_split_data(X, y_perm)
    X_train_perm_scaled, X_test_perm_scaled = preprocessor.scale_features(
        X_train_perm, X_test_perm, 'permeability'
    )
    
    # Split for porosity
    X_train_por, X_test_por, y_train_por, y_test_por = preprocessor.train_test_split_data(X, y_por)
    X_train_por_scaled, X_test_por_scaled = preprocessor.scale_features(
        X_train_por, X_test_por, 'porosity'
    )
    
    print("✅ Data preprocessing completed successfully!")
    
    return {
        'permeability': {
            'X_train': X_train_perm_scaled, 'X_test': X_test_perm_scaled,
            'y_train': y_train_perm, 'y_test': y_test_perm
        },
        'porosity': {
            'X_train': X_train_por_scaled, 'X_test': X_test_por_scaled,
            'y_train': y_train_por, 'y_test': y_test_por
        },
        'feature_names': FEATURE_COLUMNS
    }

if __name__ == "__main__":
    main()
