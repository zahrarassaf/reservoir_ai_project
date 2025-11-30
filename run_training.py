import logging
from config.data_config import DataConfig
from config.model_config import (TemporalModelConfig, PhysicsInformedConfig, 
                               EnsembleConfig)
from config.training_config import TrainingConfig
from data.spe9_loader import SPE9Loader
from core.temporal_models import TemporalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Reservoir AI Training")
    
    try:
        data_config = DataConfig()
        model_config = TemporalModelConfig()
        physics_config = PhysicsInformedConfig()
        ensemble_config = EnsembleConfig()
        training_config = TrainingConfig()
        
        data_loader = SPE9Loader(data_config)
        
        geometry = data_loader.load_grid_geometry()
        logger.info(f"Grid dimensions: {geometry['dimensions']}")
        
        static_props = data_loader.load_static_properties()
        logger.info(f"Loaded properties: {list(static_props.keys())}")
        
        model = TemporalModel(model_config)
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        features, targets = data_loader.get_training_sequences()
        logger.info(f"Training data: {features.shape} -> {targets.shape}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
