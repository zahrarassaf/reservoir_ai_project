# src/optimization/bayesian_opt.py
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np

class ESNOptimizer:
    """بهینه‌سازی پارامترهای ESN با Gaussian Process"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # فضای جستجو برای پارامترها
        self.search_space = [
            Integer(50, 2000, name='n_reservoir'),
            Real(0.5, 1.5, name='spectral_radius'),
            Real(0.01, 0.3, name='leaking_rate'),
            Real(0.05, 0.5, name='sparsity'),
            Real(1e-8, 1e-2, prior='log-uniform', name='regularization')
        ]
    
    def objective(self, params):
        """تابع هدف برای مینیمم کردن خطا"""
        n_reservoir, spectral_radius, leaking_rate, sparsity, regularization = params
        
        # ایجاد و آموزش مدل
        esn = IndustrialESN(
            n_inputs=self.X_train.shape[1],
            n_outputs=self.y_train.shape[1],
            n_reservoir=int(n_reservoir),
            spectral_radius=spectral_radius,
            leaking_rate=leaking_rate,
            sparsity=sparsity,
            regularization=regularization
        )
        
        esn.fit(self.X_train, self.y_train)
        y_pred = esn.predict(self.X_val)
        
        # محاسبه خطا (منفی NSE برای مینیمم‌سازی)
        nse = PetroleumMetrics.nash_sutcliffe(self.y_val, y_pred)
        return -nse  # مینیمم کردن منفی NSE
    
    def optimize(self, n_calls=50, random_state=42):
        """اجرای بهینه‌سازی بیزی"""
        result = gp_minimize(
            func=self.objective,
            dimensions=self.search_space,
            n_calls=n_calls,
            random_state=random_state,
            verbose=True
        )
        
        return {
            'best_params': result.x,
            'best_score': -result.fun,
            'all_results': result
        }
