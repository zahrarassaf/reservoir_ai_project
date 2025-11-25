"""
Unit tests for Reservoir AI models
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_factory import ModelFactory
from src.config import config

class TestReservoirAI(unittest.TestCase):
    """Test cases for Reservoir AI components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        
    def test_data_generation(self):
        """Test synthetic data generation"""
        df = self.data_loader.generate_spe9_synthetic_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('FlowRate', df.columns)
        self.assertIn('Pressure', df.columns)
        self.assertEqual(len(df['Well'].unique()), config.N_WELLS)
        
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        df = self.data_loader.generate_spe9_synthetic_data()
        temporal_features = self.feature_engineer.create_temporal_features(df)
        engineered_features = self.feature_engineer.create_domain_features(temporal_features)
        
        self.assertIsInstance(engineered_features, pd.DataFrame)
        self.assertGreater(len(engineered_features.columns), len(df.columns))
        
        # Check that lag features are created
        lag_columns = [col for col in engineered_features.columns if 'lag' in col]
        self.assertGreater(len(lag_columns), 0)
        
    def test_sequence_creation(self):
        """Test sequence creation for temporal models"""
        df = self.data_loader.generate_spe9_synthetic_data()
        temporal_features = self.feature_engineer.create_temporal_features(df)
        engineered_features = self.feature_engineer.create_domain_features(temporal_features)
        
        X_seq, y_seq = self.feature_engineer.prepare_sequences(engineered_features)
        
        if len(X_seq) > 0:
            self.assertEqual(len(X_seq), len(y_seq))
            self.assertEqual(X_seq.shape[1], config.SEQUENCE_LENGTH)
            
    def test_model_creation(self):
        """Test model creation from factory"""
        # Test CNN-LSTM creation
        input_shape = (10, 15)
        cnn_lstm = ModelFactory.create_cnn_lstm(input_shape)
        self.assertEqual(cnn_lstm.input_shape[1:], input_shape)
        
        # Test traditional models
        models = ModelFactory.get_all_models()
        self.assertGreater(len(models), 0)
        self.assertIn('RandomForest', models)
        self.assertIn('SVR', models)
        
    def test_data_quality(self):
        """Test data quality checks"""
        df = self.data_loader.generate_spe9_synthetic_data()
        
        # Check for NaN values
        self.assertEqual(df.isnull().sum().sum(), 0)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Pressure']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['FlowRate']))
        
    def test_feature_scaling(self):
        """Test feature scaling functionality"""
        # Generate test data
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        
        # Test scaling
        X_train_scaled, X_test_scaled = self.feature_engineer.scale_features(X_train, X_test)
        
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        
        # Check that means are approximately 0 and stds are approximately 1
        self.assertAlmostEqual(np.mean(X_train_scaled), 0, places=1)
        self.assertAlmostEqual(np.std(X_train_scaled), 1, places=1)

if __name__ == '__main__':
    unittest.main()
