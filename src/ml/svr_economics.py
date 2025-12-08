"""
Support Vector Regression for Economic Forecasting
Predicts NPV, IRR, ROI from reservoir and economic parameters
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class EconomicFeatureEngineer:
    """Feature engineering for economic prediction"""
    
    @staticmethod
    def create_features(reservoir_params: Dict[str, float], 
                       economic_params: Dict[str, float]) -> pd.DataFrame:
        """
        Create engineered features from reservoir and economic parameters
        
        Args:
            reservoir_params: Dictionary of reservoir parameters
            economic_params: Dictionary of economic parameters
        
        Returns:
            DataFrame with engineered features
        """
        
        features = {}
        
        # Basic reservoir features
        features['porosity'] = reservoir_params.get('porosity', 0.2)
        features['permeability'] = reservoir_params.get('permeability', 100)
        features['oil_in_place'] = reservoir_params.get('oil_in_place', 1e6)
        features['recoverable_oil'] = reservoir_params.get('recoverable_oil', 3e5)
        features['water_cut'] = reservoir_params.get('water_cut', 0.3)
        
        # Basic economic features
        features['oil_price'] = economic_params.get('oil_price', 60)
        features['opex_per_bbl'] = economic_params.get('opex_per_bbl', 15)
        features['capex'] = economic_params.get('capex', 20e6)
        features['discount_rate'] = economic_params.get('discount_rate', 0.1)
        features['tax_rate'] = economic_params.get('tax_rate', 0.3)
        
        # Engineered features
        # 1. Productivity indices
        features['productivity_index'] = features['permeability'] * features['porosity']
        features['recovery_factor'] = features['recoverable_oil'] / features['oil_in_place']
        
        # 2. Economic ratios
        features['price_cost_ratio'] = features['oil_price'] / features['opex_per_bbl']
        features['capex_per_bbl'] = features['capex'] / features['recoverable_oil']
        
        # 3. Time value features
        features['discounted_production'] = features['recoverable_oil'] / (1 + features['discount_rate'])
        
        # 4. Risk metrics
        features['water_risk'] = features['water_cut'] * features['opex_per_bbl']
        features['price_risk'] = features['oil_price'] * features['discount_rate']
        
        # 5. Composite metrics
        features['profitability_potential'] = (
            features['recoverable_oil'] * features['oil_price'] * 
            (1 - features['water_cut']) / features['capex']
        )
        
        # 6. Technical-economic hybrids
        features['net_pay_productivity'] = (
            features['porosity'] * features['permeability'] * 
            features['oil_price'] / features['opex_per_bbl']
        )
        
        # 7. Efficiency metrics
        features['operational_efficiency'] = (
            (1 - features['water_cut']) * features['price_cost_ratio']
        )
        
        # 8. Risk-adjusted returns
        features['risk_adjusted_npv'] = (
            features['recoverable_oil'] * features['oil_price'] *
            (1 - features['tax_rate']) / 
            (features['capex'] * (1 + features['discount_rate']))
        )
        
        # 9. Break-even features
        features['break_even_price'] = (
            features['opex_per_bbl'] + 
            (features['capex'] * features['discount_rate'] / features['recoverable_oil'])
        )
        
        # 10. Sensitivity features
        features['price_sensitivity'] = features['oil_price'] - features['break_even_price']
        features['opex_sensitivity'] = features['opex_per_bbl'] / features['oil_price']
        
        return pd.DataFrame([features])


class SVREconomicPredictor:
    """
    Advanced SVR-based economic predictor with feature engineering
    Predicts key economic indicators from reservoir parameters
    """
    
    def __init__(self, model_type='svr', use_polynomial_features=True):
        """
        Args:
            model_type: 'svr', 'random_forest', or 'gradient_boosting'
            use_polynomial_features: Whether to add polynomial features
        """
        self.model_type = model_type
        self.use_polynomial_features = use_polynomial_features
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.best_params = {}
        
        # Target variables
        self.targets = ['npv', 'irr', 'roi', 'payback_period', 'break_even_price']
    
    def prepare_data(self, X_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple:
        """
        Prepare and split data
        
        Args:
            X_data: Feature DataFrame
            y_data: Target DataFrame
        
        Returns:
            Split data
        """
        # Handle missing values
        X_data = X_data.fillna(X_data.mean())
        y_data = y_data.fillna(y_data.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self, target_name: str) -> Pipeline:
        """Create ML pipeline for a specific target"""
        
        if self.model_type == 'svr':
            base_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
            param_grid = {
                'model__C': [0.1, 1, 10, 100, 1000],
                'model__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'model__epsilon': [0.01, 0.1, 0.5]
            }
        
        elif self.model_type == 'random_forest':
            base_model = RandomForestRegressor(n_estimators=100, random_state=42)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline
        steps = [('scaler', StandardScaler())]
        
        if self.use_polynomial_features:
            steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
        
        steps.append(('model', base_model))
        pipeline = Pipeline(steps)
        
        return pipeline, param_grid
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
              cv_folds: int = 5, n_jobs: int = -1):
        """
        Train models for all targets
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Cross-validation folds
            n_jobs: Number of parallel jobs
        """
        
        print(f"Training {self.model_type.upper()} models for economic prediction...")
        print(f"Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")
        
        for target in self.targets:
            if target in y_train.columns:
                print(f"\nTraining for {target.upper()}...")
                
                # Create pipeline
                pipeline, param_grid = self.create_pipeline(target)
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline, param_grid, 
                    cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=n_jobs,
                    verbose=0
                )
                
                # Train
                grid_search.fit(X_train, y_train[target])
                
                # Store best model
                self.models[target] = grid_search.best_estimator_
                self.best_params[target] = grid_search.best_params_
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    grid_search.best_estimator_, 
                    X_train, y_train[target], 
                    cv=cv_folds, 
                    scoring='r2'
                )
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
                # Feature importance (for tree-based models)
                if hasattr(grid_search.best_estimator_.named_steps['model'], 'feature_importances_'):
                    importance = grid_search.best_estimator_.named_steps['model'].feature_importances_
                    feature_names = X_train.columns
                    
                    # Handle polynomial features
                    if self.use_polynomial_features:
                        poly = grid_search.best_estimator_.named_steps['poly']
                        feature_names = poly.get_feature_names_out(X_train.columns)
                    
                    self.feature_importance[target] = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all economic indicators"""
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(X)
        
        return pd.DataFrame(predictions, index=X.index)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        
        predictions = self.predict(X_test)
        
        metrics = {}
        for target in self.targets:
            if target in y_test.columns:
                y_true = y_test[target]
                y_pred = predictions[target]
                
                metrics[target] = {
                    'MAE': mean_absolute_error(y_true, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'R2': r2_score(y_true, y_pred),
                    'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                }
        
        return metrics
    
    def feature_analysis(self, X_train: pd.DataFrame):
        """Analyze feature importance and correlations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature correlations
        corr_matrix = X_train.corr()
        sns.heatmap(corr_matrix, ax=axes[0, 0], cmap='coolwarm', center=0,
                   square=True, cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('Feature Correlation Matrix')
        
        # 2. Feature importance for each target
        for idx, target in enumerate(self.targets[:4]):
            if target in self.feature_importance:
                ax = axes[(idx+1)//3, (idx+1)%3]
                top_features = self.feature_importance[target].head(10)
                ax.barh(top_features['feature'], top_features['importance'])
                ax.set_title(f'Top Features - {target.upper()}')
                ax.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Print top features
        print("\n" + "="*50)
        print("TOP FEATURES FOR EACH TARGET:")
        print("="*50)
        
        for target in self.targets:
            if target in self.feature_importance:
                print(f"\n{target.upper()}:")
                top5 = self.feature_importance[target].head(5)
                for _, row in top5.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def economic_sensitivity_analysis(self, base_features: pd.DataFrame, 
                                     parameter: str, 
                                     values: np.ndarray,
                                     target: str = 'npv'):
        """
        Perform sensitivity analysis for economic parameters
        
        Args:
            base_features: Base case feature vector
            parameter: Parameter to vary
            values: Array of parameter values
            target: Target variable to analyze
        """
        
        if target not in self.models:
            print(f"Model for {target} not found!")
            return
        
        sensitivities = []
        for value in values:
            # Modify parameter
            features = base_features.copy()
            if parameter in features.columns:
                features[parameter] = value
            
            # Predict
            pred = self.models[target].predict(features)[0]
            sensitivities.append(pred)
        
        # Plot sensitivity
        plt.figure(figsize=(10, 6))
        plt.plot(values, sensitivities, 'b-o', linewidth=2, markersize=8)
        plt.axhline(y=sensitivities[len(values)//2], color='r', linestyle='--', alpha=0.5)
        plt.xlabel(parameter.replace('_', ' ').title())
        plt.ylabel(target.upper())
        plt.title(f'Sensitivity Analysis: {target.upper()} vs {parameter.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Calculate sensitivity index
        base_value = sensitivities[len(values)//2]
        max_change = np.max(np.abs(np.array(sensitivities) - base_value))
        sensitivity_index = max_change / base_value * 100
        
        print(f"\nSensitivity Analysis Results:")
        print(f"Parameter: {parameter}")
        print(f"Target: {target}")
        print(f"Base {target}: ${base_value:,.2f}")
        print(f"Maximum change: ${max_change:,.2f}")
        print(f"Sensitivity Index: {sensitivity_index:.2f}%")
        
        return sensitivities
    
    def scenario_analysis(self, scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze multiple economic scenarios
        
        Args:
            scenarios: Dictionary of scenario names and feature DataFrames
        
        Returns:
            DataFrame with scenario results
        """
        
        results = {}
        for scenario_name, features in scenarios.items():
            predictions = self.predict(features)
            results[scenario_name] = predictions.mean().to_dict()
        
        return pd.DataFrame(results).T
    
    def save_model(self, path='svr_economic_model.joblib'):
        """Save the entire model"""
        model_data = {
            'models': self.models,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'targets': self.targets,
            'model_type': self.model_type,
            'use_polynomial_features': self.use_polynomial_features
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='svr_economic_model.joblib'):
        """Load saved model"""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.targets = model_data['targets']
        self.model_type = model_data['model_type']
        self.use_polynomial_features = model_data['use_polynomial_features']
        print(f"Model loaded from {path}")


# Example usage with synthetic data
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Reservoir parameters
    reservoir_data = {
        'porosity': np.random.uniform(0.1, 0.3, n_samples),
        'permeability': np.random.lognormal(3, 0.5, n_samples),
        'oil_in_place': np.random.uniform(1e6, 10e6, n_samples),
        'recoverable_oil': np.random.uniform(0.2e6, 3e6, n_samples),
        'water_cut': np.random.uniform(0.1, 0.6, n_samples)
    }
    
    # Economic parameters
    economic_data = {
        'oil_price': np.random.uniform(40, 100, n_samples),
        'opex_per_bbl': np.random.uniform(10, 30, n_samples),
        'capex': np.random.uniform(10e6, 50e6, n_samples),
        'discount_rate': np.random.uniform(0.05, 0.15, n_samples),
        'tax_rate': np.random.uniform(0.2, 0.4, n_samples)
    }
    
    # Combine features
    feature_engineer = EconomicFeatureEngineer()
    X_data = []
    
    for i in range(n_samples):
        reservoir_params = {k: v[i] for k, v in reservoir_data.items()}
        economic_params = {k: v[i] for k, v in economic_data.items()}
        features = feature_engineer.create_features(reservoir_params, economic_params)
        X_data.append(features)
    
    X = pd.concat(X_data, ignore_index=True)
    
    # Generate synthetic targets (simplified formulas)
    y_data = pd.DataFrame({
        'npv': (
            X['recoverable_oil'] * (X['oil_price'] - X['opex_per_bbl']) * 
            (1 - X['tax_rate']) / (1 + X['discount_rate']) - X['capex']
        ) / 1e6,  # in millions
        
        'irr': np.random.uniform(0.05, 0.25, n_samples) * 100,  # in percent
        
        'roi': (
            (X['recoverable_oil'] * X['oil_price'] * 0.7 - X['capex']) / X['capex']
        ) * 100,  # in percent
        
        'payback_period': np.random.uniform(1, 10, n_samples),  # in years
        
        'break_even_price': X['opex_per_bbl'] + (X['capex'] * 0.1 / X['recoverable_oil'])
    })
    
    # Create and train predictor
    predictor = SVREconomicPredictor(model_type='random_forest', use_polynomial_features=True)
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y_data)
    
    # Train
    predictor.train(X_train, y_train)
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS:")
    print("="*50)
    
    for target, target_metrics in metrics.items():
        print(f"\n{target.upper()}:")
        for metric_name, value in target_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Feature analysis
    predictor.feature_analysis(X_train)
    
    # Sensitivity analysis example
    print("\n" + "="*50)
    print("ECONOMIC SENSITIVITY ANALYSIS:")
    print("="*50)
    
    base_case = X_train.iloc[[0]]  # Use first sample as base case
    oil_prices = np.linspace(40, 100, 20)
    
    predictor.economic_sensitivity_analysis(
        base_case, 'oil_price', oil_prices, target='npv'
    )
    
    # Scenario analysis
    print("\n" + "="*50)
    print("SCENARIO ANALYSIS:")
    print("="*50)
    
    scenarios = {
        'Base Case': base_case,
        'High Price': base_case.assign(oil_price=100),
        'Low Price': base_case.assign(oil_price=40),
        'High Cost': base_case.assign(opex_per_bbl=30),
        'Low Cost': base_case.assign(opex_per_bbl=10)
    }
    
    scenario_results = predictor.scenario_analysis(scenarios)
    print("\nScenario Results:")
    print(scenario_results.round(2))
    
    # Save model
    predictor.save_model('economic_predictor_model.joblib')
