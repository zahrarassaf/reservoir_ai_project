import pytest
import torch
import torch.nn as nn

from core.temporal_models import TemporalModel

class TestTemporalModels:
    def test_lstm_initialization(self, sample_configs):
        model_config = sample_configs['model_config']
        model_config.model_type = "lstm"
        
        model = TemporalModel(model_config)
        
        assert isinstance(model.rnn, nn.LSTM)
        assert model.rnn.input_size == model_config.input_channels
        assert model.rnn.hidden_size == model_config.hidden_dim
        assert model.rnn.num_layers == model_config.num_layers
        assert model.rnn.bidirectional == model_config.bidirectional

    def test_gru_initialization(self, sample_configs):
        model_config = sample_configs['model_config']
        model_config.model_type = "gru"
        
        model = TemporalModel(model_config)
        
        assert isinstance(model.rnn, nn.GRU)
        assert model.rnn.input_size == model_config.input_channels
        assert model.rnn.hidden_size == model_config.hidden_dim

    def test_forward_pass(self, sample_configs, sample_data):
        model_config = sample_configs['model_config']
        model = TemporalModel(model_config)
        
        X, _ = sample_data
        output = model(X)
        
        # Check output shape
        assert output.shape[0] == X.shape[0]  # batch size
        assert output.shape[1] == X.shape[1]  # sequence length
        assert output.shape[2] == model_config.output_channels

    def test_attention_mechanism(self, sample_configs, sample_data):
        model_config = sample_configs['model_config']
        model_config.use_attention = True
        
        model = TemporalModel(model_config)
        X, _ = sample_data
        output = model(X)
        
        assert output.shape == (X.shape[0], X.shape[1], model_config.output_channels)
        assert model.attention is not None

    def test_physics_loss_computation(self, sample_configs, sample_data):
        model_config = sample_configs['model_config']
        model = TemporalModel(model_config)
        
        X, y = sample_data
        predictions = model(X)
        
        # Mock static properties
        static_properties = {
            'porosity': torch.randn(X.shape[0], 1),
            'permeability': torch.randn(X.shape[0], 3)
        }
        
        physics_loss = model.compute_physics_loss(predictions, y, static_properties)
        
        assert isinstance(physics_loss, torch.Tensor)
        assert physics_loss.item() >= 0.0

    def test_parameter_count(self, sample_configs):
        model_config = sample_configs['model_config']
        model = TemporalModel(model_config)
        
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_invalid_model_type(self, sample_configs):
        model_config = sample_configs['model_config']
        model_config.model_type = "invalid_type"
        
        with pytest.raises(ValueError):
            TemporalModel(model_config)
