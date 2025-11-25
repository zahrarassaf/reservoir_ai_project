"""
CNN-LSTM Model for Reservoir Forecasting
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_cnn_lstm(input_shape: tuple,
                   conv_filters: int = 32,
                   lstm_units: int = 64,
                   dense_units: int = 32,
                   dropout: float = 0.3):
    """Build CNN-LSTM model with proper architecture"""
    model = Sequential([
        Conv1D(conv_filters, kernel_size=3, activation='relu', 
               input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Dropout(dropout),
        
        LSTM(lstm_units, activation='tanh', return_sequences=False),
        Dropout(dropout),
        
        Dense(dense_units, activation='relu'),
        Dense(1)  # Single output for regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    return model

def train_cnn_lstm_model(model, X_train, y_train, X_val, y_val,
                         epochs: int = 100, batch_size: int = 32,
                         model_path: str = os.path.join(MODEL_DIR, "cnn_lstm_best.h5")):
    """Train CNN-LSTM model with callbacks"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False  # Important for time series
    )
    
    return history, model_path

def load_trained_model(path: str):
    """Load trained model"""
    return load_model(path)
