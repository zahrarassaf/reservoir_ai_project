from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def tune_svr(X_train, y_train):
    param_grid = {'C':[0.1,1,10], 'gamma':[0.01,0.1,1], 'kernel':['rbf']}
    grid = GridSearchCV(SVR(), param_grid, cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
