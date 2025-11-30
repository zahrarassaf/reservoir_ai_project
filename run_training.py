import logging
import torch
import numpy as np
from pathlib import Path

from config.data_config import DataConfig
from config.model_config import TemporalModelConfig, PhysicsInformedConfig, EnsembleConfig
from config.training_config import TrainingConfig
from data.spe9_loader import SPE9Loader
from ensemble.ensemble_trainer import EnsembleTrainer
from utils.visualization import plot_training_history, plot_reservoir_properties

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment():
    """Setup experiment configuration"""
    data_config = DataConfig()
    temporal_config = TemporalModelConfig()
    physics_config = PhysicsInformedConfig()
    ensemble_config = EnsembleConfig()
    training_config = TrainingConfig()
    
    return data_config, temporal_config, physics_config, ensemble_config, training_config

def create_data_loaders(features, targets, training_config):
    """Create PyTorch data loaders from numpy arrays"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.FloatTensor(targets)
    
    # Create dataset
    dataset = TensorDataset(features_tensor, targets_tensor)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

def main():
    try:
        logger.info("🚀 Starting Reservoir AI Training Pipeline")
        
        # Setup configurations
        data_config, temporal_config, physics_config, ensemble_config, training_config = setup_experiment()
        
        # Initialize data loader
        logger.info("📊 Loading SPE9 data...")
        data_loader = SPE9Loader(data_config)
        
        # Load and validate data
        geometry = data_loader.load_grid_geometry()
        logger.info(f"📐 Grid dimensions: {geometry['dimensions']}")
        
        static_props = data_loader.load_static_properties()
        logger.info(f"📈 Static properties: {list(static_props.keys())}")
        
        # Plot reservoir properties
        plot_reservoir_properties(static_props, layer=0)
        
        # Generate training sequences
        logger.info("🔄 Generating training sequences...")
        features, targets = data_loader.get_training_sequences()
        logger.info(f"🎯 Training data: {features.shape} -> {targets.shape}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(features, targets, training_config)
        logger.info(f"📦 Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Initialize ensemble trainer
        logger.info("🤖 Initializing ensemble trainer...")
        ensemble_trainer = EnsembleTrainer(ensemble_config, temporal_config, training_config)
        
        # Train ensemble
        logger.info("🏋️ Training ensemble models...")
        training_history = ensemble_trainer.train_ensemble(train_loader, val_loader)
        
        # Plot training history
        plot_training_history(training_history)
        
        # Save trained ensemble
        save_path = Path("checkpoints") / "trained_ensemble.pth"
        save_path.parent.mkdir(exist_ok=True)
        ensemble_trainer.save_ensemble(save_path)
        logger.info(f"💾 Ensemble saved to: {save_path}")
        
        # Final evaluation
        diversity = ensemble_trainer.compute_diversity()
        logger.info(f"🎯 Final ensemble diversity: {diversity:.4f}")
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
