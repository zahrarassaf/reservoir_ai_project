import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_svr(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svr = SVR(kernel='rbf')
    svr.fit(X_scaled, y_train)
    return svr, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
