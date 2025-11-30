import pytest
import torch
import numpy as np

from ensemble.ensemble_trainer import EnsembleTrainer

class TestEnsembleTrainer:
    def test_ensemble_initialization(self, sample_configs):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        
        assert len(trainer.models) == ensemble_config.num_models
        assert len(trainer.optimizers) == ensemble_config.num_models
        assert len(trainer.schedulers) == ensemble_config.num_models

    def test_ensemble_prediction(self, sample_configs, sample_data):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        X, _ = sample_data
        
        predictions = trainer.predict_ensemble(X)
        
        assert 'mean' in predictions
        assert 'std' in predictions
        assert 'all_predictions' in predictions
        assert predictions['mean'].shape == (X.shape[0], X.shape[1], model_config.output_channels)
        assert predictions['std'].shape == (X.shape[0], X.shape[1], model_config.output_channels)

    def test_diversity_computation(self, sample_configs):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        
        diversity = trainer.compute_diversity()
        
        assert isinstance(diversity, float)
        assert diversity >= 0.0

    def test_uncertainty_estimation(self, sample_configs, sample_data):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        X, _ = sample_data
        
        uncertainty = trainer.get_ensemble_uncertainty(X)
        
        assert 'epistemic' in uncertainty
        assert 'total' in uncertainty
        assert 'predictive_variance' in uncertainty
        assert uncertainty['epistemic'].shape == (X.shape[0], X.shape[1], model_config.output_channels)

    def test_save_load_ensemble(self, sample_configs, tmp_path):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        
        # Save ensemble
        save_path = tmp_path / "test_ensemble.pth"
        trainer.save_ensemble(save_path)
        
        # Create new trainer and load
        new_trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        new_trainer.load_ensemble(save_path)
        
        assert len(new_trainer.models) == len(trainer.models)
        # Check that parameters are loaded correctly
        for orig_model, loaded_model in zip(trainer.models, new_trainer.models):
            for orig_param, loaded_param in zip(orig_model.parameters(), loaded_model.parameters()):
                assert torch.allclose(orig_param, loaded_param)

    def test_training_history(self, sample_configs):
        ensemble_config = sample_configs['ensemble_config']
        model_config = sample_configs['model_config']
        training_config = sample_configs['training_config']
        
        trainer = EnsembleTrainer(ensemble_config, model_config, training_config)
        
        # Mock training data loader
        class MockDataLoader:
            def __iter__(self):
                return iter([(torch.randn(4, 10, 5), torch.randn(4, 10, 2)) for _ in range(3)])
        
        train_loader = MockDataLoader()
        
        history = trainer.train_ensemble(train_loader, val_loader=None)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == ensemble_config.num_models
        for model_losses in history['train_loss']:
            assert len(model_losses) == training_config.num_epochs
