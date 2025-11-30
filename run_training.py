#!/usr/bin/env python3
"""
Advanced training script for SPE9 Reservoir Ensemble Model
"""

import torch
import argparse
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import SPE9GridConfig, EnsembleModelConfig, ReservoirProperties
from src.spe9_data_parser import SPE9ProfessionalParser
from src.feature_engineer import AdvancedFeatureEngineer
from src.ensemble_model import DeepEnsembleModel
from src.ensemble_trainer import AdvancedEnsembleTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Advanced SPE9 Reservoir Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing SPE9 data')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')  # Reduced for testing
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Configuration
    grid_config = SPE9GridConfig()
    model_config = EnsembleModelConfig()
    reservoir_props = ReservoirProperties()
    
    print("🚀 Starting Advanced SPE9 Reservoir Modeling...")
    
    try:
        # 1. Load and process data
        print("📊 Loading and processing SPE9 data...")
        parser = SPE9ProfessionalParser(grid_config)
        complete_data = parser.parse_complete_spe9_system(args.data_dir)
        
        # 2. Feature engineering
        print("🔧 Engineering advanced features...")
        feature_engineer = AdvancedFeatureEngineer(grid_config)
        features = feature_engineer.create_advanced_features(complete_data)
        
        # 3. Create model
        print("🧠 Creating deep ensemble model...")
        model = DeepEnsembleModel(model_config)
        
        # 4. Training
        print("🎯 Training ensemble model...")
        trainer = AdvancedEnsembleTrainer(model, model_config)
        training_results = trainer.train_ensemble(features, args.epochs, args.batch_size)
        
        # 5. Save results
        print("💾 Saving results...")
        trainer.save_results(args.output_dir, training_results)
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
