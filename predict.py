import logging
import torch
import numpy as np
from pathlib import Path

from config.data_config import DataConfig
from config.model_config import TemporalModelConfig, EnsembleConfig
from config.training_config import TrainingConfig
from data.spe9_loader import SPE9Loader
from ensemble.ensemble_trainer import EnsembleTrainer
from utils.visualization import plot_uncertainty
from utils.metrics import reservoir_metrics, calculate_forecast_accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_ensemble(ensemble_path):
    """Load trained ensemble model"""
    ensemble_config = EnsembleConfig()
    model_config = TemporalModelConfig()
    training_config = TrainingConfig()
    
    trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
    trainer.load_ensemble(ensemble_path)
    
    return trainer

def main():
    try:
        logger.info("🔮 Starting Prediction Pipeline")
        
        # Load trained ensemble
        ensemble_path = Path("checkpoints/trained_ensemble.pth")
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Trained ensemble not found at {ensemble_path}")
        
        ensemble_trainer = load_trained_ensemble(ensemble_path)
        logger.info(f"✅ Loaded ensemble with {len(ensemble_trainer.models)} models")
        
        # Load data for prediction
        data_config = DataConfig()
        data_loader = SPE9Loader(data_config)
        
        # Generate test sequences
        features, targets = data_loader.get_training_sequences()
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        targets_tensor = torch.FloatTensor(targets)
        
        # Make predictions
        logger.info("🎯 Making ensemble predictions...")
        predictions = ensemble_trainer.predict_ensemble(features_tensor)
        
        # Calculate metrics
        metrics = reservoir_metrics(
            predictions['mean'].detach().numpy(),
            targets_tensor.numpy()
        )
        
        forecast_metrics = calculate_forecast_accuracy(
            predictions['mean'].detach().numpy(),
            targets_tensor.numpy()
        )
        
        # Log results
        logger.info("📊 Prediction Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        for metric, value in forecast_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Plot uncertainty
        plot_uncertainty(predictions, targets_tensor)
        
        # Save predictions
        output_dir = Path("predictions")
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / "predictions_mean.npy", predictions['mean'].detach().numpy())
        np.save(output_dir / "predictions_std.npy", predictions['std'].detach().numpy())
        np.save(output_dir / "targets.npy", targets_tensor.numpy())
        
        logger.info(f"💾 Predictions saved to: {output_dir}")
        logger.info("✅ Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()
