# src/reservoir_models/esn.py
import numpy as np
from scipy.sparse import random, diags
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndustrialESN:
    """
    Echo State Network برای مدلسازی مخازن نفت
    با ویژگی‌های صنعتی
    """
    
    def __init__(self, n_inputs, n_outputs, n_reservoir=1000,
                 spectral_radius=0.95, sparsity=0.1, leaking_rate=0.3,
                 regularization=1e-6, random_state=42):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Initialize weights
        self._initialize_weights()
        
        # State matrix collector
        self.states = None
        
        # Readout model
        self.readout = Ridge(alpha=regularization, random_state=random_state)
        
        # Scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
    
    def _initialize_weights(self):
        """مقداردهی وزن‌ها با تکنیک‌های پیشرفته"""
        # Win: وزن‌های ورودی
        self.Win = np.random.uniform(
            low=-1.0, high=1.0,
            size=(self.n_reservoir, self.n_inputs)
        )
        
        # W: وزن‌های مخزن (تصادفی و sparse)
        # استفاده از توزیع uniform با کنترل دقیق
        self.W = random(
            self.n_reservoir, self.n_reservoir,
            density=self.sparsity,
            random_state=self.random_state
        ).toarray()
        
        # نرمالایز کردن بر اساس شعاع طیفی
        eigenvalues = eigs(self.W, k=1, return_eigenvectors=False)
        max_eigenvalue = np.abs(eigenvalues[0])
        self.W *= self.spectral_radius / max_eigenvalue
        
        # Wback: بازخورد (اختیاری)
        self.Wback = None
        
        # Wout: در حین آموزش محاسبه می‌شود
        self.Wout = None
    
    def _compute_state(self, u, previous_state):
        """محاسبه state با معادله ESN"""
        # معادله اصلی ESN
        pre_activation = (
            np.dot(self.W, previous_state) +
            np.dot(self.Win, u)
        )
        
        # تابع فعال‌سازی (معمولاً tanh)
        new_state = np.tanh(pre_activation)
        
        # نشت
        state = (1 - self.leaking_rate) * previous_state + \
                self.leaking_rate * new_state
        
        return state
    
    def fit(self, X, y, warmup=100):
        """
        آموزش مدل
        X: shape (n_samples, n_inputs)
        y: shape (n_samples, n_outputs)
        """
        logger.info(f"Training ESN with {len(X)} samples...")
        
        # نرمال‌سازی داده‌ها
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        n_samples = X_scaled.shape[0]
        
        # جمع‌آوری states
        self.states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        
        # مرحله warmup
        for t in range(warmup):
            state = self._compute_state(X_scaled[t], state)
        
        # جمع‌آوری states برای training
        for t in range(warmup, n_samples):
            state = self._compute_state(X_scaled[t], state)
            self.states[t] = state
        
        # آموزش readout (فقط روی بخش بدون warmup)
        train_states = self.states[warmup:]
        train_targets = y_scaled[warmup:]
        
        # اضافه کردن ورودی به features (اختیاری)
        extended_features = np.hstack([
            train_states,
            X_scaled[warmup:]
        ])
        
        # آموزش رگرسیون
        self.readout.fit(extended_features, train_targets)
        self.Wout = self.readout.coef_.T
        
        logger.info("Training completed.")
        return self
    
    def predict(self, X, initial_state=None, return_states=False):
        """پیش‌بینی"""
        X_scaled = self.input_scaler.transform(X)
        n_samples = X.shape[0]
        
        states = np.zeros((n_samples, self.n_reservoir))
        predictions = np.zeros((n_samples, self.n_outputs))
        
        state = initial_state if initial_state is not None \
                else np.zeros(self.n_reservoir)
        
        for t in range(n_samples):
            state = self._compute_state(X_scaled[t], state)
            states[t] = state
            
            # ساخت feature vector
            features = np.hstack([state, X_scaled[t]])
            pred_scaled = self.readout.predict(features.reshape(1, -1))
            predictions[t] = self.output_scaler.inverse_transform(pred_scaled)
        
        if return_states:
            return predictions, states
        return predictions
