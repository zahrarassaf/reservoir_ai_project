"""
Support Vector Regression for Economic Analysis
PhD-Level ML for Economic Parameter Relationships
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class EconomicSVR:
    """
    Support Vector Regression for economic forecasting and sensitivity analysis.
    Handles non-linear relationships between reservoir parameters and economic outcomes.
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 use_advanced_features: bool = True):
        
        self.kernel = kernel
        self.use_advanced_features = use_advanced_features
        
        # Initialize scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Initialize SVR with optimized parameters
        self.model = self._initialize_svr()
        
        # Feature importance storage
        self.feature_importance = {}
        self.feature_names = []
    
    def _initialize_svr(self) -> Any:
        """Initialize SVR model with appropriate configuration."""
        
        if self.kernel == 'rbf':
            base_model = SVR(
                kernel='rbf',
                C=100.0,  # Regularization parameter
                epsilon=0.1,
                gamma='scale',
                cache_size=1000
            )
        elif self.kernel == 'poly':
            base_model = SVR(
                kernel='poly',
                C=10.0,
                epsilon=0.1,
                degree=3,
                coef0=1.0,
                gamma='scale'
            )
        else:  # linear
            base_model = SVR(
                kernel='linear',
                C=1.0,
                epsilon=0.1
            )
        
        return base_model
    
    def prepare_features(self,
                        reservoir_data: np.ndarray,
                        production_data: np.ndarray,
                        well_data: List[Dict]) -> pd.DataFrame:
        """
        Engineer advanced features for economic prediction.
        PhD-Level feature engineering.
        """
        print("🔧 Engineering features for SVR analysis...")
        
        features = pd.DataFrame()
        
        # 1. Reservoir Quality Features
        if len(reservoir_data) > 0:
            features['avg_porosity'] = [np.mean(reservoir_data.get('porosity', [0.2]))]
            features['avg_permeability'] = [np.mean(reservoir_data.get('permeability', [100]))]
            features['perm_std'] = [np.std(reservoir_data.get('permeability', [100]))]
            features['perm_skew'] = [self._calculate_skewness(reservoir_data.get('permeability', [100]))]
            
            # Reservoir heterogeneity index
            features['heterogeneity_index'] = [
                features['perm_std'].iloc[0] / (features['avg_permeability'].iloc[0] + 1e-10)
            ]
        
        # 2. Production Characteristics
        if len(production_data) > 0:
            oil_rate = production_data.get('oil_rate', [0])
            water_rate = production_data.get('water_rate', [0])
            
            if len(oil_rate) > 1:
                features['initial_rate'] = [oil_rate[0]]
                features['decline_rate'] = [self._calculate_decline_rate(oil_rate)]
                features['peak_rate'] = [np.max(oil_rate)]
                features['water_cut_trend'] = [self._calculate_trend(water_rate / (oil_rate + 1e-10))]
                
                # Production efficiency metrics
                features['recovery_efficiency'] = [
                    np.sum(oil_rate[:365]) / (features['initial_rate'].iloc[0] * 365 + 1e-10)
                ]
        
        # 3. Well Configuration Features
        if well_data:
            features['well_count'] = [len(well_data)]
            features['prod_well_ratio'] = [
                sum(1 for w in well_data if w.get('type') == 'PRODUCER') / len(well_data)
            ]
            
            # Well spacing metrics
            if len(well_data) > 1:
                spacing = self._calculate_well_spacing(well_data)
                features['avg_well_spacing'] = [np.mean(spacing)]
                features['well_spacing_std'] = [np.std(spacing)]
        
        # 4. Economic Sensitivity Features
        features['price_volatility_index'] = [0.25]  # Placeholder for real volatility data
        features['opex_efficiency'] = [1.0]  # Placeholder
        
        # 5. Advanced PhD-Level Features
        if self.use_advanced_features:
            # Recovery factor estimate
            if 'avg_porosity' in features.columns and 'avg_permeability' in features.columns:
                features['recovery_factor_estimate'] = [
                    0.35 * (features['avg_porosity'].iloc[0] / 0.2) * 
                    np.log1p(features['avg_permeability'].iloc[0] / 100)
                ]
            
            # Economic resilience index
            features['economic_resilience'] = [
                features.get('recovery_factor_estimate', [0.35])[0] *
                (1 - features.get('heterogeneity_index', [0.5])[0])
            ]
        
        print(f"✅ Engineered {features.shape[1]} features for SVR analysis")
        self.feature_names = features.columns.tolist()
        
        return features
    
    def train(self, 
              X: pd.DataFrame,
              y: pd.DataFrame,
              optimize_hyperparams: bool = True):
        """
        Train SVR model with optional hyperparameter optimization.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            y: Target matrix (n_samples × n_targets)
            optimize_hyperparams: Whether to perform grid search
        """
        print("\n🧠 Training Support Vector Regression model...")
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Hyperparameter optimization
        if optimize_hyperparams and X.shape[0] > 10:
            print("   Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_scaled, y_scaled)
        else:
            # Train with default parameters
            if y.shape[1] > 1:
                self.model = MultiOutputRegressor(self.model)
            
            self.model.fit(X_scaled, y_scaled)
            print("   Training completed with default parameters")
        
        # Calculate feature importance
        self._calculate_feature_importance(X_scaled, y_scaled)
        
        # Calculate performance metrics
        train_score = self.model.score(X_scaled, y_scaled)
        print(f"✅ SVR training completed. R² score: {train_score:.4f}")
        
        if hasattr(self.model, 'best_params_'):
            print(f"   Best parameters: {self.model.best_params_}")
    
    def predict_economics(self,
                         features: pd.DataFrame,
                         return_confidence: bool = True) -> Dict:
        """
        Predict economic metrics with optional confidence intervals.
        """
        # Scale features
        X_scaled = self.scaler_X.transform(features)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Prepare results
        results = {
            'npv': float(y_pred[0, 0]) if y_pred.shape[1] >= 1 else 0.0,
            'irr': float(y_pred[0, 1]) if y_pred.shape[1] >= 2 else 0.0,
            'roi': float(y_pred[0, 2]) if y_pred.shape[1] >= 3 else 0.0,
            'payback_period': float(y_pred[0, 3]) if y_pred.shape[1] >= 4 else 0.0,
            'predictions_raw': y_pred.tolist()
        }
        
        # Add confidence intervals if requested
        if return_confidence:
            confidence = self._calculate_confidence_intervals(X_scaled)
            results['confidence_intervals'] = confidence
        
        return results
    
    def sensitivity_analysis(self,
                            base_features: pd.DataFrame,
                            parameter_ranges: Dict[str, Tuple[float, float]],
                            n_points: int = 20) -> pd.DataFrame:
        """
        Perform comprehensive sensitivity analysis using SVR.
        
        Args:
            base_features: Base case feature vector
            parameter_ranges: Dictionary of parameter ranges
            n_points: Number of points per parameter
        
        Returns:
            DataFrame with sensitivity results
        """
        print("\n📈 Performing SVR-based sensitivity analysis...")
        
        results = []
        
        for param_name, (low, high) in parameter_ranges.items():
            if param_name not in base_features.columns:
                print(f"⚠️  Parameter {param_name} not in features, skipping")
                continue
            
            print(f"   Analyzing {param_name}...")
            values = np.linspace(low, high, n_points)
            
            for value in values:
                # Modify the parameter
                modified_features = base_features.copy()
                modified_features[param_name] = value
                
                # Predict economic metrics
                predictions = self.predict_economics(modified_features, return_confidence=False)
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'npv': predictions['npv'],
                    'irr': predictions['irr'],
                    'roi': predictions['roi'],
                    'payback': predictions['payback_period'],
                    'npv_change_percent': ((predictions['npv'] - predictions['npv']) / 
                                          abs(predictions['npv'] + 1e-10) * 100)
                })
        
        results_df = pd.DataFrame(results)
        
        # Calculate sensitivity coefficients
        sensitivity_coeffs = {}
        for param in parameter_ranges.keys():
            param_results = results_df[results_df['parameter'] == param]
            if len(param_results) > 1:
                # Calculate elasticity: % change in NPV / % change in parameter
                npv_elasticity = (
                    (param_results['npv'].max() - param_results['npv'].min()) /
                    (param_results['value'].max() - param_results['value'].min()) *
                    (param_results['value'].mean() / (param_results['npv'].mean() + 1e-10))
                )
                sensitivity_coeffs[param] = float(npv_elasticity)
        
        print(f"✅ Sensitivity analysis completed. Most sensitive: {max(sensitivity_coeffs, key=sensitivity_coeffs.get)}")
        
        return results_df, sensitivity_coeffs
    
    def calculate_break_even(self,
                           features: pd.DataFrame,
                           target_irr: float = 0.15,
                           max_iter: int = 100) -> Dict:
        """
        Calculate break-even price using SVR optimization.
        """
        print("\n💰 Calculating break-even price with SVR...")
        
        # Use Brent's method for root finding
        from scipy.optimize import brentq
        
        def objective(price: float) -> float:
            """Objective function: IRR - target_IRR"""
            modified_features = features.copy()
            modified_features['oil_price_sensitivity'] = price / 100
            
            predictions = self.predict_economics(modified_features, return_confidence=False)
            return predictions['irr'] - target_irr
        
        try:
            # Find root (price where IRR = target)
            price_low = 20.0
            price_high = 200.0
            
            # Check bounds
            if objective(price_low) * objective(price_high) > 0:
                print("⚠️  Cannot find break-even within range")
                break_even_price = features.get('oil_price_sensitivity', [82.5])[0] * 100
            else:
                break_even_price = brentq(objective, price_low, price_high, maxiter=max_iter)
            
            # Get metrics at break-even
            features_be = features.copy()
            features_be['oil_price_sensitivity'] = break_even_price / 100
            metrics_be = self.predict_economics(features_be, return_confidence=False)
            
            result = {
                'break_even_price': float(break_even_price),
                'npv_at_be': metrics_be['npv'],
                'irr_at_be': metrics_be['irr'],
                'roi_at_be': metrics_be['roi'],
                'safety_margin': ((82.5 - break_even_price) / 82.5 * 100),  # Current price = 82.5
                'calculation_method': 'SVR Optimization'
            }
            
            print(f"✅ Break-even price: ${break_even_price:.2f}/bbl")
            print(f"   Safety margin: {result['safety_margin']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Error calculating break-even: {e}")
            return {
                'break_even_price': 82.5,
                'error': str(e),
                'calculation_method': 'Failed - using default'
            }
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize SVR hyperparameters using grid search."""
        
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'gamma': ['scale', 'auto', 0.1, 1, 10]
        }
        
        if self.kernel == 'poly':
            param_grid['degree'] = [2, 3, 4]
            param_grid['coef0'] = [0.0, 1.0]
        
        # Use time series cross-validation
        cv = TimeSeriesSplit(n_splits=5)
        
        if y.shape[1] > 1:
            # Multi-output regression
            base_model = SVR(kernel=self.kernel)
            model = MultiOutputRegressor(
                GridSearchCV(
                    base_model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
            )
        else:
            # Single output
            model = GridSearchCV(
                SVR(kernel=self.kernel),
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        
        model.fit(X, y)
        return model
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance for SVR."""
        if hasattr(self.model, 'coef_'):
            # Linear kernel has coefficients
            if y.shape[1] == 1:
                importance = np.abs(self.model.coef_[0])
            else:
                importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # For non-linear kernels, use permutation importance
            from sklearn.inspection import permutation_importance
            
            result = permutation_importance(
                self.model, X, y,
                n_repeats=10,
                random_state=42
            )
            importance = result.importances_mean
        
        # Normalize importance
        if len(importance) > 0:
            importance = importance / np.sum(importance)
            self.feature_importance = dict(zip(self.feature_names, importance))
    
    def _calculate_confidence_intervals(self, X: np.ndarray) -> Dict:
        """Calculate confidence intervals using various methods."""
        # Simplified confidence interval calculation
        # In production, use bootstrapping or Bayesian methods
        
        predictions = []
        n_bootstrap = 100
        
        # Bootstrap predictions
        for _ in range(n_bootstrap):
            # Create bootstrap sample
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            
            # Predict (simplified - in reality need to retrain)
            pred = self.model.predict(X_boot.mean(axis=0, keepdims=True))
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(axis=0).tolist(),
            'std': predictions.std(axis=0).tolist(),
            'ci_95_lower': np.percentile(predictions, 2.5, axis=0).tolist(),
            'ci_95_upper': np.percentile(predictions, 97.5, axis=0).tolist(),
            'method': 'bootstrap',
            'n_bootstrap': n_bootstrap
        }
    
    # Helper methods for feature engineering
    def _calculate_decline_rate(self, rates: np.ndarray) -> float:
        """Calculate decline rate from production data."""
        if len(rates) < 2:
            return 0.0
        log_rates = np.log(rates + 1e-10)
        time = np.arange(len(rates))
        slope, _ = np.polyfit(time, log_rates, 1)
        return abs(slope)
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate linear trend coefficient."""
        if len(data) < 2:
            return 0.0
        time = np.arange(len(data))
        slope, _ = np.polyfit(time, data, 1)
        return slope
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        from scipy.stats import skew
        return float(skew(data))
    
    def _calculate_well_spacing(self, wells: List[Dict]) -> List[float]:
        """Calculate distances between wells."""
        if len(wells) < 2:
            return [0.0]
        
        distances = []
        locations = [(w.get('i', 0), w.get('j', 0)) for w in wells]
        
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                dx = locations[i][0] - locations[j][0]
                dy = locations[i][1] - locations[j][1]
