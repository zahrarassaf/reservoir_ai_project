"""
Comprehensive tests for model implementations.
"""

import pytest
import numpy as np
from typing import Dict, Any
import warnings

from src.models.esn import EchoStateNetwork, ESNConfig
from src.models.deep_esn import DeepEchoStateNetwork, DeepESNConfig
from src.models.advanced_esn import (
    HierarchicalESN, HierarchicalESNConfig,
    AttentionESN, AttentionESNConfig,
    PhysicsInformedESN, PhysicsInformedESNConfig
)


class TestEchoStateNetwork:
    """Test suite for EchoStateNetwork."""
    
    def test_initialization(self, esn_config):
        """Test ESN initialization."""
        esn = EchoStateNetwork(esn_config)
        
        # Check weight matrices shapes
        assert esn.W_in.shape == (esn_config.n_reservoir, esn_config.n_inputs)
        assert esn.W_res.shape == (esn_config.n_reservoir, esn_config.n_reservoir)
        
        # Check spectral radius scaling
        eigenvalues = np.linalg.eigvals(esn.W_res)
        spectral_radius = np.max(np.abs(eigenvalues))
        assert np.isclose(spectral_radius, esn_config.spectral_radius, rtol=0.1)
        
        # Check sparsity
        sparsity = np.mean(esn.W_res == 0)
        assert np.isclose(sparsity, 1 - esn_config.sparsity, rtol=0.1)
    
    def test_different_configurations(self):
        """Test ESN with various configurations."""
        configs = [
            ESNConfig(n_inputs=5, n_outputs=2, n_reservoir=200,
                     spectral_radius=0.8, sparsity=0.05),
            ESNConfig(n_inputs=3, n_outputs=1, n_reservoir=500,
                     reservoir_connectivity="small_world"),
            ESNConfig(n_inputs=8, n_outputs=4, n_reservoir=300,
                     activation_function="relu"),
            ESNConfig(n_inputs=6, n_outputs=3, n_reservoir=150,
                     teacher_forcing=True),
        ]
        
        for config in configs:
            esn = EchoStateNetwork(config)
            
            # Should initialize without errors
            assert esn is not None
            
            # Check basic properties
            assert esn.W_in.shape == (config.n_reservoir, config.n_inputs)
            assert esn.config == config
    
    def test_training_basic(self, esn_config, synthetic_data):
        """Test basic training functionality."""
        esn = EchoStateNetwork(esn_config)
        X = synthetic_data['X'][:500]
        y = synthetic_data['y'][:500]
        
        # Train model
        stats = esn.fit(X, y)
        
        # Check that training completed
        assert stats is not None
        assert isinstance(stats, dict)
        assert 'training_metrics' in stats
        
        # Check that model learned something
        assert esn.W_out is not None
        assert esn.W_out.shape == (esn_config.n_reservoir + esn_config.n_inputs, 
                                  esn_config.n_outputs)
    
    def test_prediction_shape(self, trained_esn, synthetic_data):
        """Test prediction output shape."""
        X_test = synthetic_data['X'][800:900]
        
        # Test basic prediction
        y_pred = trained_esn.predict(X_test)
        
        assert y_pred.shape == (100, trained_esn.config.n_outputs)
        
        # Test prediction with states
        y_pred, states = trained_esn.predict(X_test, return_states=True)
        
        assert y_pred.shape == (100, trained_esn.config.n_outputs)
        assert states.shape == (100, trained_esn.config.n_reservoir)
    
    def test_prediction_consistency(self, trained_esn, synthetic_data):
        """Test that predictions are consistent across multiple calls."""
        X_test = synthetic_data['X'][800:850]
        
        # Get predictions twice
        y_pred1 = trained_esn.predict(X_test)
        y_pred2 = trained_esn.predict(X_test)
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(y_pred1, y_pred2, decimal=10)
    
    def test_memory_property(self, esn_config, synthetic_data):
        """Test memory property of ESN."""
        esn = EchoStateNetwork(esn_config)
        X = synthetic_data['X'][:300]
        y = synthetic_data['y'][:300]
        
        # Train on sequence
        esn.fit(X, y)
        
        # Test memory by feeding delayed version
        delay = 10
        X_delayed = np.roll(X, delay, axis=0)
        
        # Predict with delayed input
        y_pred_delayed = esn.predict(X_delayed)
        
        # Compare with original predictions (shifted)
        y_pred_original = esn.predict(X)
        
        # Delayed predictions should correlate with shifted original predictions
        correlation = np.corrcoef(
            y_pred_delayed[delay:].flatten(),
            y_pred_original[:-delay].flatten()
        )[0, 1]
        
        # ESN should have some memory
        assert correlation > 0.3
    
    def test_serialization(self, trained_esn, temp_dir):
        """Test model serialization and deserialization."""
        # Save model
        save_path = temp_dir / "test_esn.pkl"
        trained_esn.save(save_path)
        assert save_path.exists()
        
        # Load model
        loaded_esn = EchoStateNetwork.load(save_path)
        
        # Compare configurations
        assert loaded_esn.config.__dict__ == trained_esn.config.__dict__
        
        # Compare weight matrices
        np.testing.assert_array_almost_equal(loaded_esn.W_in, trained_esn.W_in)
        np.testing.assert_array_almost_equal(loaded_esn.W_res, trained_esn.W_res)
        
        # Compare predictions
        test_input = np.random.randn(10, trained_esn.config.n_inputs)
        pred_original = trained_esn.predict(test_input)
        pred_loaded = loaded_esn.predict(test_input)
        
        np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=10)
    
    def test_teacher_forcing(self, synthetic_data):
        """Test teacher forcing functionality."""
        config_with_tf = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=100,
            teacher_forcing=True,
            random_state=42
        )
        
        config_without_tf = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=100,
            teacher_forcing=False,
            random_state=42
        )
        
        X = synthetic_data['X'][:400]
        y = synthetic_data['y'][:400]
        
        # Train both models
        esn_tf = EchoStateNetwork(config_with_tf)
        esn_no_tf = EchoStateNetwork(config_without_tf)
        
        stats_tf = esn_tf.fit(X, y)
        stats_no_tf = esn_no_tf.fit(X, y)
        
        # Both should train successfully
        assert 'training_metrics' in stats_tf
        assert 'training_metrics' in stats_no_tf
        
        # Test predictions
        X_test = synthetic_data['X'][400:500]
        y_pred_tf = esn_tf.predict(X_test)
        y_pred_no_tf = esn_no_tf.predict(X_test)
        
        assert y_pred_tf.shape == (100, 1)
        assert y_pred_no_tf.shape == (100, 1)
    
    def test_noise_injection(self):
        """Test noise injection during training."""
        config_with_noise = ESNConfig(
            n_inputs=5,
            n_outputs=1,
            n_reservoir=100,
            noise_level=0.1,
            random_state=42
        )
        
        config_without_noise = ESNConfig(
            n_inputs=5,
            n_outputs=1,
            n_reservoir=100,
            noise_level=0.0,
            random_state=42
        )
        
        # Generate simple data
        X = np.random.randn(200, 5)
        y = np.random.randn(200, 1)
        
        esn_noise = EchoStateNetwork(config_with_noise)
        esn_no_noise = EchoStateNetwork(config_without_noise)
        
        # Both should train
        esn_noise.fit(X, y)
        esn_no_noise.fit(X, y)
        
        # Noise should add some stochasticity during training
        # (but predictions are still deterministic)
        X_test = np.random.randn(10, 5)
        pred_noise = esn_noise.predict(X_test)
        pred_no_noise = esn_no_noise.predict(X_test)
        
        # Predictions should be different due to different training
        assert not np.allclose(pred_noise, pred_no_noise)
    
    def test_state_reset(self, trained_esn, synthetic_data):
        """Test reservoir state reset functionality."""
        X_test = synthetic_data['X'][800:810]
        
        # Get initial prediction
        y_pred1, states1 = trained_esn.predict(X_test, return_states=True)
        
        # Reset state
        trained_esn.reset_state()
        
        # Get prediction again
        y_pred2, states2 = trained_esn.predict(X_test, return_states=True)
        
        # First states should be different (reset vs. continuation)
        assert not np.allclose(states1[0], states2[0])
        
        # But predictions should eventually converge
        np.testing.assert_array_almost_equal(y_pred1[-1], y_pred2[-1], decimal=5)
    
    def test_error_handling(self, esn_config):
        """Test error handling for invalid inputs."""
        esn = EchoStateNetwork(esn_config)
        
        # Wrong input dimension
        X_wrong = np.random.randn(10, esn_config.n_inputs + 1)
        y = np.random.randn(10, 1)
        
        with pytest.raises(AssertionError):
            esn.fit(X_wrong, y)
        
        # Wrong output dimension
        X = np.random.randn(10, esn_config.n_inputs)
        y_wrong = np.random.randn(10, esn_config.n_outputs + 1)
        
        with pytest.raises(AssertionError):
            esn.fit(X, y_wrong)
        
        # NaN values
        X_nan = X.copy()
        X_nan[5, 2] = np.nan
        
        with pytest.raises(ValueError):
            esn.fit(X_nan, y)
        
        # Infinite values
        X_inf = X.copy()
        X_inf[5, 2] = np.inf
        
        with pytest.raises(ValueError):
            esn.fit(X_inf, y)
    
    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
    def test_activation_functions(self, activation, synthetic_data):
        """Test different activation functions."""
        config = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=50,
            activation_function=activation,
            random_state=42
        )
        
        esn = EchoStateNetwork(config)
        X = synthetic_data['X'][:200]
        y = synthetic_data['y'][:200]
        
        # Should train without errors
        stats = esn.fit(X, y)
        assert 'training_metrics' in stats
        
        # Should make predictions
        X_test = synthetic_data['X'][200:250]
        y_pred = esn.predict(X_test)
        assert y_pred.shape == (50, 1)
    
    @pytest.mark.parametrize("connectivity", ["uniform", "small_world", "scale_free"])
    def test_connectivity_patterns(self, connectivity, synthetic_data):
        """Test different reservoir connectivity patterns."""
        config = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=100,
            reservoir_connectivity=connectivity,
            random_state=42
        )
        
        esn = EchoStateNetwork(config)
        X = synthetic_data['X'][:300]
        y = synthetic_data['y'][:300]
        
        # Train
        stats = esn.fit(X, y)
        assert 'training_metrics' in stats
        
        # Check connectivity statistics
        if connectivity == "uniform":
            density = np.mean(esn.W_res != 0)
            assert np.isclose(density, config.sparsity, rtol=0.2)
        
        # Test predictions
        X_test = synthetic_data['X'][300:350]
        y_pred = esn.predict(X_test)
        assert y_pred.shape == (50, 1)
    
    def test_performance_on_chaotic_data(self, chaotic_data):
        """Test ESN performance on chaotic time series."""
        config = ESNConfig(
            n_inputs=2,
            n_outputs=1,
            n_reservoir=200,
            spectral_radius=0.9,
            leaking_rate=0.3,
            random_state=42
        )
        
        esn = EchoStateNetwork(config)
        X = chaotic_data['X'][:800]
        y = chaotic_data['y'][:800]
        
        # Train on chaotic data
        stats = esn.fit(X, y)
        
        # Test short-term prediction
        X_test = chaotic_data['X'][800:900]
        y_test = chaotic_data['y'][800:900]
        
        y_pred = esn.predict(X_test)
        
        # Calculate prediction error
        mse = np.mean((y_test - y_pred) ** 2)
        variance = np.var(y_test)
        
        # Should capture some dynamics
        nse = 1 - mse / variance
        assert nse > 0.5  # Reasonable performance on chaotic data
    
    def test_large_reservoir(self):
        """Test ESN with large reservoir size."""
        config = ESNConfig(
            n_inputs=5,
            n_outputs=1,
            n_reservoir=1000,  # Large reservoir
            sparsity=0.01,     # Very sparse
            random_state=42
        )
        
        esn = EchoStateNetwork(config)
        
        # Generate data
        X = np.random.randn(500, 5)
        y = np.random.randn(500, 1)
        
        # Should train without memory issues
        stats = esn.fit(X, y)
        assert 'training_metrics' in stats
        
        # Check sparsity
        density = np.mean(esn.W_res != 0)
        assert np.isclose(density, config.sparsity, rtol=0.2)
    
    def test_regularization_effect(self, synthetic_data):
        """Test effect of regularization."""
        configs = [
            ESNConfig(n_inputs=10, n_outputs=1, n_reservoir=100,
                     regularization=1e-8, random_state=42),  # Low regularization
            ESNConfig(n_inputs=10, n_outputs=1, n_reservoir=100,
                     regularization=1e-2, random_state=42),  # High regularization
        ]
        
        X = synthetic_data['X'][:400]
        y = synthetic_data['y'][:400]
        X_val = synthetic_data['X'][400:500]
        y_val = synthetic_data['y'][400:500]
        
        errors = []
        
        for config in configs:
            esn = EchoStateNetwork(config)
            esn.fit(X, y)
            
            y_pred = esn.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            errors.append(mse)
        
        # High regularization should prevent overfitting
        # (simplified test - in practice need proper validation)
        assert errors[0] > 0  # Low regularization
        assert errors[1] > 0  # High regularization


class TestDeepEchoStateNetwork:
    """Test suite for DeepEchoStateNetwork."""
    
    def test_initialization(self, deep_esn_config):
        """Test Deep ESN initialization."""
        deep_esn = DeepEchoStateNetwork(deep_esn_config)
        
        # Check number of layers
        assert len(deep_esn.layers) == deep_esn_config.n_layers
        assert len(deep_esn.level_readouts) == deep_esn_config.n_layers
        
        # Check layer sizes
        for i, (layer, expected_size) in enumerate(
            zip(deep_esn.layers, deep_esn_config.layer_sizes)
        ):
            assert layer.config.n_reservoir == expected_size
            
            if i == 0:
                assert layer.config.n_inputs == deep_esn_config.n_inputs
            else:
                assert layer.config.n_inputs == deep_esn_config.layer_sizes[i-1]
    
    def test_training(self, deep_esn_config, synthetic_data):
        """Test Deep ESN training."""
        deep_esn = DeepEchoStateNetwork(deep_esn_config)
        X = synthetic_data['X'][:600]
        y = synthetic_data['y'][:600]
        
        # Train
        stats = deep_esn.fit(X, y)
        
        # Check training completed
        assert stats is not None
        assert isinstance(stats, dict)
        
        # Check layer statistics
        for i in range(deep_esn_config.n_layers):
            assert f'layer_{i}' in stats
        
        # Check final readout trained
        assert 'final' in stats
        assert 'training_metrics' in stats['final']
    
    def test_prediction(self, deep_esn_config, synthetic_data):
        """Test Deep ESN predictions."""
        deep_esn = DeepEchoStateNetwork(deep_esn_config)
        X_train = synthetic_data['X'][:600]
        y_train = synthetic_data['y'][:600]
        X_test = synthetic_data['X'][600:700]
        
        # Train
        deep_esn.fit(X_train, y_train)
        
        # Test basic prediction
        y_pred = deep_esn.predict(X_test)
        assert y_pred.shape == (100, deep_esn_config.n_outputs)
        
        # Test prediction with layer outputs
        y_pred, layer_outputs = deep_esn.predict(X_test, return_layer_outputs=True)
        
        assert y_pred.shape == (100, deep_esn_config.n_outputs)
        assert len(layer_outputs) == deep_esn_config.n_layers
        
        for i, output in enumerate(layer_outputs):
            assert output.shape == (100, deep_esn_config.layer_sizes[i])
    
    def test_aggregation_methods(self, synthetic_data):
        """Test different aggregation methods."""
        methods = ["concatenate", "average", "weighted"]
        
        for method in methods:
            config = DeepESNConfig(
                n_inputs=10,
                n_outputs=1,
                n_layers=2,
                layer_sizes=[50, 50],
                aggregation_method=method
            )
            
            deep_esn = DeepEchoStateNetwork(config)
            X = synthetic_data['X'][:500]
            y = synthetic_data['y'][:500]
            
            # Should train without errors
            stats = deep_esn.fit(X, y)
            assert 'final' in stats
            
            # Should predict
            X_test = synthetic_data['X'][500:550]
            y_pred = deep_esn.predict(X_test)
            assert y_pred.shape == (50, 1)
    
    def test_skip_connections(self, synthetic_data):
        """Test skip connections in Deep ESN."""
        config_with_skip = DeepESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_layers=2,
            layer_sizes=[50, 50],
            use_skip_connections=True
        )
        
        config_without_skip = DeepESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_layers=2,
            layer_sizes=[50, 50],
            use_skip_connections=False
        )
        
        X = synthetic_data['X'][:400]
        y = synthetic_data['y'][:400]
        X_test = synthetic_data['X'][400:450]
        
        # Train both
        deep_esn_skip = DeepEchoStateNetwork(config_with_skip)
        deep_esn_no_skip = DeepEchoStateNetwork(config_without_skip)
        
        stats_skip = deep_esn_skip.fit(X, y)
        stats_no_skip = deep_esn_no_skip.fit(X, y)
        
        # Both should train
        assert 'final' in stats_skip
        assert 'final' in stats_no_skip
        
        # Both should predict
        y_pred_skip = deep_esn_skip.predict(X_test)
        y_pred_no_skip = deep_esn_no_skip.predict(X_test)
        
        assert y_pred_skip.shape == (50, 1)
        assert y_pred_no_skip.shape == (50, 1)
    
    def test_get_layer(self, deep_esn_config):
        """Test get_layer method."""
        deep_esn = DeepEchoStateNetwork(deep_esn_config)
        
        # Get each layer
        for i in range(deep_esn_config.n_layers):
            layer = deep_esn.get_layer(i)
            assert layer is not None
            assert isinstance(layer, EchoStateNetwork)
            assert layer.config.n_reservoir == deep_esn_config.layer_sizes[i]
        
        # Test invalid index
        with pytest.raises(IndexError):
            deep_esn.get_layer(deep_esn_config.n_layers)


class TestAdvancedESNModels:
    """Test suite for advanced ESN variants."""
    
    def test_hierarchical_esn(self, synthetic_data):
        """Test HierarchicalESN."""
        base_config = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=100,
            random_state=42
        )
        
        hier_config = HierarchicalESNConfig(
            base_config=base_config,
            n_levels=3,
            reduction_factor=0.5
        )
        
        hier_esn = HierarchicalESN(hier_config)
        X = synthetic_data['X'][:500]
        y = synthetic_data['y'][:500]
        
        # Train
        stats = hier_esn.fit(X, y)
        
        # Check structure
        assert len(hier_esn.levels) == hier_config.n_levels
        assert len(hier_esn.level_readouts) == hier_config.n_levels
        
        # Check level sizes decrease
        sizes = [level.config.n_reservoir for level in hier_esn.levels]
        assert all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))
        
        # Test prediction
        X_test = synthetic_data['X'][500:550]
        y_pred = hier_esn.predict(X_test)
        assert y_pred.shape == (50, 1)
    
    def test_attention_esn(self, synthetic_data):
        """Test AttentionESN."""
        base_config = ESNConfig(
            n_inputs=10,
            n_outputs=1,
            n_reservoir=100,
            random_state=42
        )
        
        attn_config = AttentionESNConfig(
            base_config=base_config,
            attention_mechanism="self_attention",
            n_attention_heads=4
        )
        
        attn_esn = AttentionESN(attn_config)
        X = synthetic_data['X'][:400]
        y = synthetic_data['y'][:400]
        
        # Train
        stats = attn_esn.fit(X, y)
        
        # Check attention weights were stored
        assert len(attn_esn.attention_weights) > 0
        
        # Check attention statistics
        assert 'attention_stats' in stats
        assert 'attention_entropy_mean' in stats['attention_stats']
        
        # Test prediction
        X_test = synthetic_data['X'][400:450]
        y_pred = attn_esn.predict(X_test)
        assert y_pred.shape == (50, 1)
    
    def test_physics_informed_esn(self):
        """Test PhysicsInformedESN."""
        base_config = ESNConfig(
            n_inputs=2,
            n_outputs=1,
            n_reservoir=100,
            random_state=42
        )
        
        physics_constraints = {
            "material_balance": {
                "compressibility": 1e-5,
                "volume": 1.0
            },
            "boundary_conditions": {
                "left_value": 3000,
                "right_value": 3000
            }
        }
        
        physics_config = PhysicsInformedESNConfig(
            base_config=base_config,
            physics_constraints=physics_constraints,
            constraint_weight=0.1
        )
        
        physics_esn = PhysicsInformedESN(physics_config)
        
        # Generate synthetic reservoir data
        n_samples = 200
        time = np.linspace(0, 100, n_samples)
        pressure = 3000 + 500 * np.sin(time / 10)
        production = 100 * np.ones(n_samples)
        
        X = np.column_stack([time[:-1], production[:-1]])
        y = pressure[1:].reshape(-1, 1)
        
        additional_data = {
            "pressure": pressure,
            "production": production,
            "compressibility": 1e-5,
            "volume": 1.0,
            "dt": time[1] - time[0],
            "time": time
        }
        
        # Train with physics constraints
        stats = physics_esn.fit(X, y, additional_data)
        
        # Check physics loss was computed
        assert "initial_physics_loss" in stats
        
        # Test prediction
        X_test = np.column_stack([
            np.linspace(100, 110, 10),
            100 * np.ones(10)
        ])
        y_pred = physics_esn.predict(X_test)
        assert y_pred.shape == (10, 1)
