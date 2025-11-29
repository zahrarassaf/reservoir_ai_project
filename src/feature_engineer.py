import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AdvancedFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.feature_selector = None
        self.pca = None
        
    def create_advanced_features(self, X, feature_names):
        """Create sophisticated feature engineering pipeline"""
        print("\n🔧 ADVANCED FEATURE ENGINEERING")
        print("===============================")
        
        original_features = X.shape[-1]
        engineered_features = []
        
        # 1. Rolling statistics
        X_with_rolling = self._add_rolling_features(X, feature_names)
        engineered_features.extend([f"rolling_{name}" for name in feature_names])
        
        # 2. Temporal features
        X_with_time = self._add_temporal_features(X_with_rolling)
        
        # 3. Interaction features
        X_with_interactions = self._add_interaction_features(X_with_time, feature_names)
        
        # 4. Statistical features
        X_with_stats = self._add_statistical_features(X_with_interactions)
        
        print(f"   Original features: {original_features}")
        print(f"   Engineered features: {X_with_stats.shape[-1]}")
        
        return X_with_stats
    
    def _add_rolling_features(self, X, feature_names):
        """Add rolling window statistics"""
        rolling_features = []
        
        for window in self.config.ROLLING_WINDOWS:
            if window < X.shape[1]:
                # Rolling mean
                roll_mean = np.nanmean(self._rolling_window(X, window, axis=1), axis=2)
                # Rolling std
                roll_std = np.nanstd(self._rolling_window(X, window, axis=1), axis=2)
                
                rolling_features.extend([roll_mean, roll_std])
        
        if rolling_features:
            return np.concatenate([X] + rolling_features, axis=-1)
        return X
    
    def _add_temporal_features(self, X):
        """Add temporal characteristics"""
        batch_size, seq_len, n_features = X.shape
        
        # Position encoding
        positions = np.arange(seq_len) / seq_len
        positions = np.tile(positions, (batch_size, n_features, 1)).transpose(0, 2, 1)
        
        # Seasonal patterns
        time_of_cycle = np.sin(2 * np.pi * np.arange(seq_len) / 365)
        time_of_cycle = np.tile(time_of_cycle, (batch_size, n_features, 1)).transpose(0, 2, 1)
        
        return np.concatenate([X, positions, time_of_cycle], axis=-1)
    
    def _add_interaction_features(self, X, feature_names):
        """Add feature interactions"""
        # This would be implemented based on domain knowledge
        # For now, return original features
        return X
    
    def _add_statistical_features(self, X):
        """Add statistical moment features"""
        batch_size, seq_len, n_features = X.shape
        
        # Statistical moments along sequence axis
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        skewness = self._skewness(X, axis=1, keepdims=True)
        
        statistical_features = np.concatenate([mean, std, skewness], axis=-1)
        statistical_features = np.repeat(statistical_features, seq_len, axis=1)
        
        return np.concatenate([X, statistical_features], axis=-1)
    
    def _rolling_window(self, a, window, axis=1):
        """Create rolling window view of array"""
        shape = a.shape[:axis] + (a.shape[axis] - window + 1, window) + a.shape[axis+1:]
        strides = a.strides[:axis] + (a.strides[axis], a.strides[axis]) + a.strides[axis+1:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def _skewness(self, a, axis=None, keepdims=False):
        """Calculate skewness along specified axis"""
        a = np.asarray(a)
        mean = np.mean(a, axis=axis, keepdims=True)
        std = np.std(a, axis=axis, keepdims=True, ddof=0)
        skew = np.mean(((a - mean) / std) ** 3, axis=axis, keepdims=keepdims)
        return skew
    
    def select_features(self, X, y, k=20):
        """Select most important features"""
        print(f"\n🎯 FEATURE SELECTION (top {k} features)")
        
        # Reshape for feature selection
        X_flat = X.reshape(-1, X.shape[-1])
        y_repeated = np.repeat(y, X.shape[1])
        
        # Use mutual information for non-linear relationships
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[-1]))
        X_selected = self.feature_selector.fit_transform(X_flat, y_repeated)
        
        # Reshape back to sequences
        X_selected = X_selected.reshape(X.shape[0], X.shape[1], -1)
        
        print(f"   Selected {X_selected.shape[-1]} features from {X.shape[-1]}")
        
        return X_selected
