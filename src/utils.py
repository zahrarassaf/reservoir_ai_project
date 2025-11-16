"""
hyperparameter_tuning.py
Small wrappers to run GridSearch for SVR; Keras Tuner optional for CNN-LSTM.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def tune_svr(X_train_scaled, y_train_scaled, cv=3, n_jobs=-1):
    param_grid = {'C':[1,10,50], 'epsilon':[0.01,0.1,0.5], 'kernel':['rbf']}
    grid = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs)
    grid.fit(X_train_scaled, y_train_scaled)
    return grid.best_estimator_, grid.best_params_
