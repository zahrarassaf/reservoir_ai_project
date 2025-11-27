"""
PRODUCTION TEST SUITE
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
    """COMPREHENSIVE TEST SUITE"""
    
    def setUp(self):
        """SETUP TEST ENVIRONMENT"""
        self.loader = ReservoirDataLoader()
        self.feature_engineer = ReservoirFeatureEngineer()
    
    def test_data_loading(self):
        """TEST DATA LOADING FUNCTIONALITY"""
        data = self.loader.load_data()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('oil_rate', data.columns)
        self.assertIn('bottomhole_pressure', data.columns)
        
        print("✅ DATA LOADING TEST PASSED")
    
    def test_feature_engineering(self):
        """TEST FEATURE ENGINEERING PIPELINE"""
        data = self.loader.load_data()
        X, y, features, engineered_data = self.feature_engineer.prepare_features(data)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertGreater(len(features), 0)
        
        print("✅ FEATURE ENGINEERING TEST PASSED")
    
    def test_model_initialization(self):
        """TEST MODEL INITIALIZATION"""
        model = AdvancedReservoirModel()
        self.assertIsNotNone(model)
        
        # TEST WITH SAMPLE DATA
        data = self.loader.load_data()
        X, y, features, _ = self.feature_engineer.prepare_features(data)
        
        if len(X) > 0:
            X_flat = X.reshape(X.shape[0], -1)
            model.train_ensemble(X_flat[:100], y[:100])  # Small subset
            
            self.assertTrue(model.is_trained)
        
        print("✅ MODEL INITIALIZATION TEST PASSED")

def run_tests():
    """RUN TEST SUITE"""
    print("🧪 RUNNING RESERVOIR AI TEST SUITE...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
