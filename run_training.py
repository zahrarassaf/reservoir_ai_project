#!/usr/bin/env python3

import torch
import argparse
from pathlib import Path
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import (SPE9GridConfig, ReservoirPhysicsConfig, 
                               TemporalModelConfig, PhysicsInformedConfig, EnsembleConfig)
from src.data_loader import SPE9DataLoader
from src.feature_engineer import PhysicsAwareFeatureEngineer
from src.ensemble_model import DiverseReservoirModel
from src.physics_informed import PhysicsInformedLoss
from src.training_orchestrator import ProfessionalTrainer

def setup_experiment(args):
    """Setup experiment configuration and directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = {
        'timestamp': timestamp,
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    with open(experiment_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
    return experiment_dir

def main():
    parser = argparse.ArgumentParser(description='Professional SPE9 Reservoir Modeling')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing SPE9 data files')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--use_physics', action='store_true',
                       help='Enable physics-informed training')
    
    args = parser.parse_args()
    
    # Setup experiment
    experiment_dir = setup_experiment(args)
    print(f"🚀 Starting professional SPE9 reservoir modeling experiment...")
    print(f"📁 Experiment directory: {experiment_dir}")
    
    try:
        # Configuration
        grid_config = SPE9GridConfig()
        physics_config = ReservoirPhysicsConfig()
        temporal_config = TemporalModelConfig()
        physics_informed_config = PhysicsInformedConfig(use_physics_loss=args.use_physics)
        ensemble_config = EnsembleConfig()
        
        # 1. Load real SPE9 data
        print("📊 Loading real SPE9 dataset...")
        data_loader = SPE9DataLoader(grid_config)
        reservoir_data = data_loader.load_complete_dataset(args.data_dir)
        
        print(f"✅ Loaded reservoir data:")
        print(f"   - Grid: {reservoir_data.permeability.shape}")
        print(f"   - Wells: {len(reservoir_data.well_locations)}")
        print(f"   - Production history: {len(reservoir_data.production['FOPR'])} days")
        
        # 2. Professional feature engineering
        print("🔧 Engineering physics-aware features...")
        feature_engineer = PhysicsAwareFeatureEngineer(grid_config, physics_config)
        features = feature_engineer.create_advanced_features(reservoir_data)
        
        print(f"✅ Created {len(features)} feature groups")
        
        # 3. Create diverse ensemble model
        print("🧠 Creating diverse ensemble model...")
        model_config = TemporalModelConfig()  # Using temporal config as base
        ensemble_model = DiverseReservoirModel(model_config, ensemble_config)
        
        print(f"✅ Ensemble created with {len(ensemble_model.models)} diverse models")
        
        # 4. Physics-informed loss
        physics_loss = PhysicsInformedLoss(physics_informed_config, grid_config)
        
        # 5. Professional training
        print("🎯 Starting professional training...")
        trainer = ProfessionalTrainer(
            model=ensemble_model,
            physics_loss=physics_loss,
            experiment_dir=experiment_dir
        )
        
        # Prepare training data
        training_data = trainer.prepare_training_data(reservoir_data, features)
        
        # Train with proper validation and testing
        results = trainer.train(
            training_data=training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_physics=args.use_physics
        )
        
        # 6. Comprehensive evaluation
        print("📈 Running comprehensive evaluation...")
        evaluation_results = trainer.evaluate_comprehensive(results, reservoir_data)
        
        # 7. Save professional report
        trainer.save_professional_report(results, evaluation_results)
        
        print("✅ Professional training completed successfully!")
        print(f"📊 Results saved to: {experiment_dir}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_log = experiment_dir / "error_log.txt"
        with open(error_log, 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    main()
