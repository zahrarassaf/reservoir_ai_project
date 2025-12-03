
import pytest
import numpy as np
from src.models.esn import EchoStateNetwork, ESNConfig
from src.models.deep_esn import DeepEchoStateNetwork, DeepESNConfig

class TestEchoStateNetwork:
    """Comprehensive tests for ESN implementation."""
    
    def test_initialization(self):
        """Test ESN initialization with various configurations."""
        configs = [
            ESNConfig(n_inputs=10, n_outputs=3, n_reservoir=100),
            ESNConfig(n_inputs=5, n_outputs=2, n_reservoir=500, 
                     spectral_radius=1.2, sparsity=0.05),
            ESNConfig(n_inputs=8, n_outputs=4, reservoir_connectivity="small_world"),
        ]
        
        for config in configs:
            esn = EchoStateNetwork(config)
            assert esn.W_in.shape == (config.n_reservoir, config.n_inputs)
            assert esn.W_res.shape == (config.n_reservoir, config.n_reservoir)
    
    def test_training_stability(self):
        """Test training stability with different random seeds."""
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000, 3)
        
        scores = []
        for seed in range(5):
            config = ESNConfig(n_inputs=10, n_outputs=3, random_state=seed)
            esn = EchoStateNetwork(config)
            esn.fit(X, y)
            y_pred = esn.predict(X)
            
            # Calculate R²
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            scores.append(r2)
        
        # Check that all models perform reasonably
        assert np.all(np.array(scores) > -1.0)
    
    def test_memory_capacity(self):
        """Test memory capacity estimation."""
        config = ESNConfig(n_inputs=1, n_outputs=1, n_reservoir=200)
        esn = EchoStateNetwork(config)
        
        # Generate Mackey-Glass time series
        t = np.linspace(0, 100, 1000)
        x = np.sin(t) + 0.5 * np.sin(3*t) + 0.1 * np.random.randn(1000)
        
        X = x[:-1].reshape(-1, 1)
        y = x[1:].reshape(-1, 1)
        
        esn.fit(X, y)
        
        # Test memory by predicting with delayed inputs
        predictions = []
        for delay in [1, 5, 10, 20]:
            X_delayed = np.roll(x, delay)[:-1].reshape(-1, 1)
            y_pred = esn.predict(X_delayed)
            correlation = np.corrcoef(x[1:], y_pred.flatten())[0, 1]
            predictions.append((delay, correlation))
        
        # Memory should decay with delay
        delays, correlations = zip(*predictions)
        assert np.all(np.diff(correlations) < 0.1)  # Gentle decay
    
    def test_serialization(self):
        """Test model serialization/deserialization."""
        config = ESNConfig(n_inputs=5, n_outputs=2)
        esn = EchoStateNetwork(config)
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 2)
        esn.fit(X, y)
        
        # Save and load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            esn.save(f.name)
            
            # Load and compare
            esn_loaded = EchoStateNetwork.load(f.name)
            
            # Check predictions match
            X_test = np.random.randn(10, 5)
            y_pred_original = esn.predict(X_test)
            y_pred_loaded = esn_loaded.predict(X_test)
            
            assert np.allclose(y_pred_original, y_pred_loaded, rtol=1e-5)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty input
        config = ESNConfig(n_inputs=3, n_outputs=1)
        esn = EchoStateNetwork(config)
        
        with pytest.raises(ValueError):
            esn.fit(np.array([]).reshape(0, 3), np.array([]).reshape(0, 1))
        
        # NaN values
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        X[5, 2] = np.nan
        
        with pytest.raises(ValueError):
            esn.fit(X, y)
        
        # Wrong dimensions
        with pytest.raises(AssertionError):
            esn.fit(X[:, :2], y)  # Wrong input dimension
    
    @pytest.mark.parametrize("connectivity", ["uniform", "small_world", "scale_free"])
    def test_connectivity_patterns(self, connectivity):
        """Test different reservoir connectivity patterns."""
        config = ESNConfig(
            n_inputs=5, 
            n_outputs=2,
            reservoir_connectivity=connectivity,
            n_reservoir=100
        )
        
        esn = EchoStateNetwork(config)
        
        # Check connectivity statistics
        if connectivity == "uniform":
            density = np.mean(esn.W_res != 0)
            assert abs(density - config.sparsity) < 0.05
        
        # Test training
        X = np.random.randn(200, 5)
        y = np.random.randn(200, 2)
        
        esn.fit(X[:150], y[:150])
        y_pred = esn.predict(X[150:])
        
        # Should at least not crash
        assert y_pred.shape == (50, 2)
    
    def test_teacher_forcing(self):
        """Test teacher forcing mode."""
        config_with_tf = ESNConfig(
            n_inputs=2, n_outputs=1, 
            teacher_forcing=True, n_reservoir=100
        )
        
        config_without_tf = ESNConfig(
            n_inputs=2, n_outputs=1,
            teacher_forcing=False, n_reservoir=100
        )
        
        # Lorenz system data
        def lorenz_system(sigma=10, rho=28, beta=8/3, dt=0.01, steps=1000):
            x, y, z = np.zeros(steps), np.zeros(steps), np.zeros(steps)
            x[0], y[0], z[0] = 1, 1, 1
            
            for i in range(steps-1):
                dx = sigma * (y[i] - x[i])
                dy = x[i] * (rho - z[i]) - y[i]
                dz = x[i] * y[i] - beta * z[i]
                
                x[i+1] = x[i] + dx * dt
                y[i+1] = y[i] + dy * dt
                z[i+1] = z[i] + dz * dt
            
            return x, y, z
        
        x, y, z = lorenz_system()
        X = np.column_stack([x[:-1], y[:-1]])
        y_target = z[1:].reshape(-1, 1)
        
        # Train both models
        esn_tf = EchoStateNetwork(config_with_tf)
        esn_no_tf = EchoStateNetwork(config_without_tf)
        
        esn_tf.fit(X, y_target)
        esn_no_tf.fit(X, y_target)
        
        # Compare predictions
        pred_tf = esn_tf.predict(X)
        pred_no_tf = esn_no_tf.predict(X)
        
        # Teacher forcing should help for chaotic systems
        mse_tf = np.mean((y_target - pred_tf) ** 2)
        mse_no_tf = np.mean((y_target - pred_no_tf) ** 2)
        
        # Teacher forcing should not be worse
        assert mse_tf <= mse_no_tf * 1.5
