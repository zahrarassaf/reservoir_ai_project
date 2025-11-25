"""
Integration tests for complete pipeline
"""
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import main
from src.config import config

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def test_complete_pipeline(self):
        """Test complete training pipeline"""
        try:
            # Run main training function
            results = main()
            
            # Check that results are returned
            self.assertIsInstance(results, dict)
            self.assertIn('performance_summary', results)
            self.assertIn('comprehensive_report', results)
            
            # Check that output files are created
            self.assertTrue((config.RESULT_DIR / 'model_performance_summary.csv').exists())
            self.assertTrue((config.RESULT_DIR / 'comprehensive_evaluation_report.csv').exists())
            
        except Exception as e:
            self.fail(f"Pipeline failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
