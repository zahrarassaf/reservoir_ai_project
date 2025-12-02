# src/utils/metrics.py
import numpy as np
from scipy import stats

class PetroleumMetrics:
    """معیارهای ارزیابی خاص صنعت نفت"""
    
    @staticmethod
    def nash_sutcliffe(y_true, y_pred):
        """Nash-Sutcliffe Efficiency (NSE)"""
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else -np.inf
    
    @staticmethod
    def r2_eff(y_true, y_pred):
        """R² efficiency"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
    
    @staticmethod
    def mape(y_true, y_pred, epsilon=1e-10):
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def material_balance_error(pressure_pred, production_pred, compressibility):
        """خطای بالانس ماده (مهم در شبیه‌سازی مخزن)"""
        # محاسبه خطای بالانس ماده
        pass
    
    @staticmethod
    def forecast_skill_score(y_true, y_pred, y_baseline):
        """مهارت پیش‌بینی نسبت به baseline"""
        mse_pred = np.mean((y_true - y_pred) ** 2)
        mse_base = np.mean((y_true - y_baseline) ** 2)
        return 1 - (mse_pred / mse_base)
