"""
COMPREHENSIVE TEST SUITE FOR RESERVOIR AI MODELS
PRODUCTION-READY TESTING
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import ReservoirDataLoader
from src.feature_engineer import ReservoirFeatureEngineer
from src.ensemble_model import AdvancedReservoirModel
from src.config import config

class TestReservoirAI(unittest.TestCase):
    """COMPREHENSIVE TEST SUITE FOR RESERVOIR AI"""
    
    def setUp(self):
        """SETUP TEST ENVIRONMENT"""
        self.loader = ReservoirDataLoader()
        self.feature_engineer = ReservoirFeatureEngineer()
        self.model = AdvancedReservoirModel()
    
    def test_data_loader(self):
        """TEST DATA LOADER FUNCTIONALITY"""
        data = self.loader.generate_physics_based_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('oil_rate', data.columns)
        self.assertIn('bottomhole_pressure', data.columns)
        
        print("✅ DATA LOADER TEST PASSED")
    
    def test_feature_engineering(self):
        """TEST FEATURE ENGINEERING PIPELINE"""
        data = self.loader.generate_physics_based_data()
        X, y, features, engineered_data = self.feature_engineer.prepare_features(data)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertGreater(len(features), 0)
        self.assertEqual(X.shape[0], y.shape[0])
        
        print("✅ FEATURE ENGINEERING TEST PASSED")
    
    def test_model_initialization(self):
        """TEST MODEL INITIALIZATION"""
        self.assertIsNotNone(self.model)
        
        # TEST MODEL BUILDING
        input_shape = (45, 15)  # sequence_length, num_features
        model = self.model.build_hybrid_cnn_lstm(input_shape)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 1))
        
        print("✅ MODEL INITIALIZATION TEST PASSED")
    
    def test_prediction_shape(self):
        """TEST PREDICTION SHAPES"""
        # GENERATE TEST DATA
        data = self.loader.generate_physics_based_data()
        X, y, features, _ = self.feature_engineer.prepare_features(data)
        
        # TEST WITH SMALL SUBSET
        X_test = X[:10]
        X_flat = X_test.reshape(X_test.shape[0], -1)
        
        # BUILD AND TEST MODEL
        self.model.build_ml_ensemble()
        predictions = self.model.predict_ensemble(X_test, X_flat)
        
        self.assertIsInstance(predictions, dict)
        self.assertGreater(len(predictions), 0)
        
        for model_name, pred in predictions.items():
            self.assertEqual(pred.shape, (10,))
        
        print("✅ PREDICTION SHAPE TEST PASSED")

def run_tests():
    """RUN COMPLETE TEST SUITE"""
    print("🧪 RUNNING RESERVOIR AI TEST SUITE...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
