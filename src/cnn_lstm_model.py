"""
cnn_lstm_model.py
Builds, trains and evaluates a CNN-LSTM model for per-well forecasting.
"""

from typing import Tuple
import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_cnn_lstm(input_shape: Tuple[int,int],
                   conv_filters: int = 32,
                   lstm_units: int = 64,
                   dense_units: int = 64,
                   dropout: float = 0.2):
    model = Sequential()
    model.add(Conv1D(conv_filters, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_units, activation='tanh', return_sequences=False))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(input_shape[0]))  # predict one flow value per well
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_cnn_lstm_model(model, X_train, y_train, X_val, y_val,
                         epochs: int = 100, batch_size: int = 8,
                         model_path: str = os.path.join(MODEL_DIR, "cnn_lstm_best.h5")):
    """
    Train model with early stopping and model checkpoint. Returns history and path to best model.
    """
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    mc = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[es, mc], verbose=2)
    return history, model_path

def load_trained_model(path: str):
    return load_model(path)
