"""
Support Vector Regression for Economic Analysis and Sensitivity
PhD-Level Implementation for Non-linear Regression
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from typing import Tuple, Dict, List, Optional

class EconomicSVR:
    """
    Support Vector Regression for economic parameter relationships.
    Handles non-linear relationships between reservoir parameters and economic outcomes.
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 use_grid_search: bool = True):
        
        self.kernel = kernel
        self.use_grid_search = use_grid_search
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        if use_grid_search:
            self.model = self._create_grid_search_model()
        else:
            self.model = SVR(
                kernel=kernel,
                C=1.0,
                epsilon=0.1,
                gamma='scale'
            )
    
    def _create_grid_search_model(self) -> GridSearchCV:
        """Create SVR with grid search for hyperparameter optimization."""
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'gamma': ['scale', 'auto', 0.1, 1, 10]
        }
        
        if self.kernel == 'poly':
            param_grid['degree'] = [2, 3, 4]
            param_grid['coef0'] = [0.0, 1.0]
        
        cv = TimeSeriesSplit(n_splits=5)
        
        return GridSearchCV(
            SVR(kernel=self.kernel),
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
    
    def prepare_economic_data(self,
                            reservoir_data: pd.DataFrame,
                            economic_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for SVR training."""
        
        # Feature engineering
        features = self._engineer_features(reservoir_data)
        
        # Economic targets
        targets = economic_data[['npv', 'irr', 'roi', 'payback_period']].values
        
        # Handle missing values
        features = pd.DataFrame(features).fillna(method='ffill').fillna(0).values
        targets = pd.DataFrame(targets).fillna(method='ffill').fillna(0).values
        
        return features, targets
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for economic prediction."""
        
        features = pd.DataFrame()
        
        # Basic reservoir features
        features['porosity_mean'] = data['porosity'].mean(axis=1)
        features['permeability_mean'] = data['permeability'].mean(axis=1)
        features['net_thickness'] = data['thickness'].sum(axis=1)
        
        # Production-related features
        features['initial_rate'] = data['production'].apply(lambda x: x.iloc[0] if len(x) > 0 else 0)
        features['decline_rate'] = self._calculate_decline_rate(data['production'])
        features['cumulative_180'] = data['production'].apply(lambda x: x.iloc[:180].sum())
        
        # Spatial features
        features['well_spacing'] = self._calculate_well_spacing(data['well_locations'])
        features['reservoir_continuity'] = self._calculate_continuity(data['permeability'])
        
        # Economic sensitivity features
        features['oil_price_sensitivity'] = data['oil_price'] / 100
        features['opex_sensitivity'] = data['operating_cost'] / data['oil_price']
        
        return features
    
    def train_multi_target(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          test_size: float = 0.2):
        """Train SVR for multiple economic targets."""
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train-test split with time series consideration
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Train model
        if y.shape[1] > 1:
            # Multi-output regression
            self.model = MultiOutputRegressor(self.model)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training R²: {train_score:.4f}")
        print(f"Testing R²: {test_score:.4f}")
        
        if self.use_grid_search and hasattr(self.model, 'best_params_'):
            print(f"Best parameters: {self.model.best_params_}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'model': self.model
        }
    
    def predict_economic_metrics(self,
                               reservoir_params: Dict[str, float]) -> Dict[str, float]:
        """Predict economic metrics from reservoir parameters."""
        
        # Convert to feature vector
        features = self._params_to_features(reservoir_params)
        features_scaled = self.scaler_X.transform([features])
        
        # Predict
        predictions_scaled = self.model.predict(features_scaled)
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions_scaled)[0]
        
        return {
            'npv': predictions[0],
            'irr': predictions[1],
            'roi': predictions[2],
            'payback_period': predictions[3]
        }
    
    def sensitivity_analysis(self,
                            base_params: Dict[str, float],
                            parameter_ranges: Dict[str, Tuple[float, float]],
                            n_points: int = 20) -> pd.DataFrame:
        """Perform comprehensive sensitivity analysis using SVR."""
        
        results = []
        
        for param_name, (low, high) in parameter_ranges.items():
            values = np.linspace(low, high, n_points)
            
            for value in values:
                # Modify parameter
                modified_params = base_params.copy()
                modified_params[param_name] = value
                
                # Predict economic metrics
                predictions = self.predict_economic_metrics(modified_params)
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'npv': predictions['npv'],
                    'irr': predictions['irr'],
                    'roi': predictions['roi'],
                    'payback': predictions['payback_period']
                })
        
        return pd.DataFrame(results)
    
    def calculate_break_even(self,
                           reservoir_params: Dict[str, float],
                           target_irr: float = 0.15) -> Dict[str, float]:
        """Calculate break-even prices using SVR optimization."""
        
        from scipy.optimize import minimize
        
        def objective(price):
            params = reservoir_params.copy()
            params['oil_price'] = price[0]
            predictions = self.predict_economic_metrics(params)
            return (predictions['irr'] - target_irr) ** 2
        
        # Initial guess
        initial_price = reservoir_params.get('oil_price', 80.0)
        
        # Optimize
        result = minimize(objective, [initial_price], bounds=[(20, 150)])
        
        break_even_price = result.x[0]
        
        # Calculate other metrics at break-even
        params_be = reservoir_params.copy()
        params_be['oil_price'] = break_even_price
        metrics_be = self.predict_economic_metrics(params_be)
        
        return {
            'break_even_price': break_even_price,
            'npv_at_be': metrics_be['npv'],
            'irr_at_be': metrics_be['irr'],
            'roi_at_be': metrics_be['roi'],
            'optimization_success': result.success
        }
    
    def feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance for economic predictions."""
        
        if hasattr(self.model, 'coef_'):
            # Linear kernel
            importance = np.abs(self.model.coef_)
        elif self.kernel == 'rbf':
            # Use permutation importance
            importance = self._calculate_permutation_importance()
        else:
            importance = np.ones(self.scaler_X.n_features_in_)
        
        features = [f'feature_{i}' for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance,
            'importance_normalized': importance / importance.sum()
        }).sort_values('importance', ascending=False)
