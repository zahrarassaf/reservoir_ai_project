import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Input, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

class ReservoirModelFactory:
    def __init__(self, config):
        self.config = config
    
    def create_cnn_lstm_model(self, input_shape):
        """Create optimized CNN-LSTM architecture"""
        model = Sequential([
            Input(shape=input_shape),
            
            # First Conv Block
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second Conv Block
            Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # LSTM Encoder
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # LSTM Decoder
            LSTM(50, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense Layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        
        # Custom optimizer configuration
        optimizer = Adam(
            learning_rate=self.config.CNN_LSTM_CONFIG['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber_loss',  # More robust than MSE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_transformer_model(self, input_shape, num_heads=4, ff_dim=64):
        """Experimental Transformer architecture for time series"""
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = self._positional_encoding(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[-1] // num_heads
        )(x, x)
        
        x = LayerNormalization()(x + attention_output)
        
        # Feed forward
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dense(input_shape[-1])(ff_output)
        x = LayerNormalization()(x + ff_output)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber_loss',
            metrics=['mae']
        )
        
        return model
    
    def _positional_encoding(self, inputs):
        """Add positional encoding to inputs"""
        seq_len = inputs.shape[1]
        d_model = inputs.shape[2]
        
        positions = np.arange(seq_len)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, dtype=tf.float32)
    
    def create_random_forest(self):
        """Create optimized Random Forest model"""
        return RandomForestRegressor(**self.config.RANDOM_FOREST_PARAMS)
    
    def create_xgboost(self):
        """Create optimized XGBoost model"""
        return XGBRegressor(**self.config.XGBOOST_PARAMS)
    
    def create_lightgbm(self):
        """Create optimized LightGBM model"""
        return LGBMRegressor(**self.config.LIGHTGBM_PARAMS)
    
    def get_callbacks(self, model_name):
        """Get advanced training callbacks"""
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                min_delta=self.config.MIN_DELTA,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/best_{model_name}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
