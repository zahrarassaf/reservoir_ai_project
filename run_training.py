#!/usr/bin/env python3

import torch
import argparse
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import SPE9GridConfig, EnsembleModelConfig
from src.spe9_data_parser import SPE9DataParser
from src.feature_engineer import FeatureEngineer
from src.ensemble_model import DeepEnsembleModel
from src.ensemble_trainer import EnsembleTrainer

def main():
    parser = argparse.ArgumentParser(description='Train SPE9 Reservoir Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing SPE9 data')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    grid_config = SPE9GridConfig()
    model_config = EnsembleModelConfig()
    
    print("Starting SPE9 Reservoir Modeling...")
    
    try:
        print("Loading SPE9 data...")
        spe9_parser = SPE9DataParser(grid_config)
        
        spe9_paths = [
            os.path.join(args.data_dir, "SPE9.DATA"),
            os.path.join(args.data_dir, "SPE9_CP.DATA"),
            os.path.join(args.data_dir, "SPE9_CP_GROUP.DATA")
        ]
        
        df = None
        for path in spe9_paths:
            if os.path.exists(path):
                print(f"Found SPE9 file: {path}")
                df = spe9_parser.parse_spe9_data(path)
                break
        
        if df is None:
            print("SPE9 files not found, using synthetic data")
            from src.opm_data_loader import OPMDataLoader
            opm_loader = OPMDataLoader(grid_config)
            df = opm_loader.load_opm_data()
        
        print("Engineering features...")
        feature_engineer = FeatureEngineer(grid_config)
        features = feature_engineer.create_features(df)
        
        print("Creating ensemble model...")
        model = DeepEnsembleModel(model_config)
        
        print("Training model...")
        trainer = EnsembleTrainer(model, model_config)
        training_results = trainer.train_ensemble(features, args.epochs, args.batch_size)
        
        print("Saving results...")
        trainer.save_results(args.output_dir, training_results)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
