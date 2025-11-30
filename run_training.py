import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.data_config import DataConfig
from data.spe9_loader import SPE9Loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting Reservoir AI with REAL SPE9 Data")
        
        # Use REAL data configuration
        config = DataConfig(development_mode=False)
        
        # Initialize REAL data loader
        data_loader = SPE9Loader(config)
        
        # Load REAL data
        geometry = data_loader.load_grid_geometry()
        logger.info(f"REAL Grid: {geometry['dimensions']}")
        
        static_props = data_loader.load_static_properties()
        logger.info(f"REAL Properties: {list(static_props.keys())}")
        
        # Show REAL data statistics
        for prop_name, data in static_props.items():
            logger.info(f"  {prop_name}: {data.shape} "
                       f"[{data.min():.3f}, {data.max():.3f}]")
        
        # Generate training data
        features, targets = data_loader.get_training_sequences()
        logger.info(f"Training data: {features.shape} -> {targets.shape}")
        
        logger.info("REAL data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load REAL data: {e}")
        raise

if __name__ == "__main__":
    main()
