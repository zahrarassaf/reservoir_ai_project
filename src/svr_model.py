"""
svr_model.py
Helpers to train and evaluate SVR baseline on flattened per-well features.
"""

from typing import Tuple, Dict
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_svr(X_train: np.ndarray, y_train: np.ndarray,
              C: float = 10.0, epsilon: float = 0.1, kernel: str = 'rbf'):
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
    Xs = scaler_X.transform(X_train)
    ys = scaler_y.transform(y_train.reshape(-1,1)).ravel()
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr.fit(Xs, ys)
    return {"model": svr, "scaler_X": scaler_X, "scaler_y": scaler_y}

def evaluate_svr(trained: dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    scaler_X = trained["scaler_X"]
    scaler_y = trained["scaler_y"]
    model = trained["model"]
    Xs = scaler_X.transform(X_test)
    y_pred_scaled = model.predict(Xs)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "r2": r2, "y_pred": y_pred}
