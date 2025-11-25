"""
Professional Model Training Module
Advanced ensemble methods with comprehensive hyperparameter optimization
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ProfessionalModelTrainer:
    """
    Professional model trainer with advanced techniques
    Includes ensemble methods, hyperparameter optimization, and model persistence
    """
    
    def __init__(self):
        self.trained_models = {}
        self.best_models = {}
        self.cv_results = {}
        self.final_ensemble = None
        
    def get_model_configurations(self):
        """Define comprehensive model configurations"""
        
        base_models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=config.RANDOM_STATE),
                'params': config.RF_PARAMS,
                'search_method': 'grid'
            },
            'XGBoost': {
                'model': XGBRegressor(random_state=config.RANDOM_STATE, verbosity=0),
                'params': config.XGB_PARAMS,
                'search_method': 'grid'
            },
            'SVM': {
                'model': SVR(),
                'params': config.SVM_PARAMS,
                'search_method': 'grid'
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=config.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5]
                },
                'search_method': 'grid'
            },
            'LightGBM': {
                'model': LGBMRegressor(random_state=config.RANDOM_STATE, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 50]
                },
                'search_method': 'random'
            },
            'Ridge Regression': {
                'model': Ridge(random_state=config.RANDOM_STATE),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'search_method': 'grid'
            }
        }
        
        return base_models
    
    def hyperparameter_tuning(self, X_train, y_train, model_name, model_config):
        """Advanced hyperparameter optimization"""
        
        if model_config['search_method'] == 'grid':
            search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                model_config['model'],
                model_config['params'],
                n_iter=10,
                cv=config.CV_FOLDS,
                scoring='r2',
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
                verbose=0
            )
        
        print(f"   🔧 Tuning {model_name}...")
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_score_, search.best_params_
    
    def create_advanced_ensemble(self, base_models, X_train, y_train):
        """Create advanced ensemble using stacking"""
        
        # First level: base models
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Second level: meta-model
        meta_model = LinearRegression()
        
        # Create stacking ensemble
        stacking_ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=config.CV_FOLDS,
            n_jobs=-1
        )
        
        # Train stacking ensemble
        stacking_ensemble.fit(X_train, y_train)
        
        return stacking_ensemble
    
    def create_voting_ensemble(self, base_models, X_train, y_train):
        """Create voting ensemble"""
        estimators = [(name, model) for name, model in base_models.items()]
        voting_ensemble = VotingRegressor(estimators=estimators)
        voting_ensemble.fit(X_train, y_train)
        
        return voting_ensemble
    
    def comprehensive_cross_validation(self, model, X_train, y_train, model_name):
        """Comprehensive cross-validation with multiple metrics"""
        
        cv_results = {}
        
        for metric in config.REGRESSION_METRICS:
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=config.CV_FOLDS, 
                scoring=metric,
                n_jobs=-1
            )
            cv_results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        # Print CV results
        r2_mean = cv_results['r2']['mean']
        r2_std = cv_results['r2']['std']
        print(f"   📊 CV R²: {r2_mean:.4f} (±{r2_std:.4f})")
        
        return cv_results
    
    def evaluate_model_performance(self, model, X_sets, y_sets, model_name, feature_names):
        """Comprehensive model evaluation on all datasets"""
        
        results = {}
        
        for set_name, (X, y) in X_sets.items():
            if X is not None and y is not None:
                y_pred = model.predict(X)
                
                metrics = {
                    'r2': r2_score(y, y_pred),
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred))
                }
                
                # Calculate additional metrics
                mape = np.mean(np.abs((y - y_pred) / np.clip(np.abs(y), 1e-8, None))) * 100
                metrics['mape'] = mape
                
                results[set_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'actual': y.values if hasattr(y, 'values') else y
                }
                
                print(f"   {set_name.upper()} R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return results
    
    def save_trained_model(self, model, model_name, target_name):
        """Save trained model with proper naming"""
        filename = config.MODELS_DIR / f"{target_name}_{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        print(f"   💾 Saved: {filename.name}")
    
    def train_models_for_target(self, datasets, target_name):
        """Complete training pipeline for a target variable"""
        
        print(f"\n🎯 Training models for: {target_name}")
        print("=" * 50)
        
        # Get data
        X_train = datasets[target_name]['X_train']
        X_val = datasets[target_name]['X_val']
        X_test = datasets[target_name]['X_test']
        y_train = datasets[target_name]['y_train']
        y_val = datasets[target_name]['y_val']
        y_test = datasets[target_name]['y_test']
        feature_names = datasets[target_name]['feature_names']
        
        # Prepare data for evaluation
        X_sets = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Get model configurations
        model_configs = self.get_model_configurations()
        base_models = {}
        training_results = {}
        
        # Train individual models
        for model_name, config in model_configs.items():
            print(f"\n🤖 Training {model_name}...")
            
            try:
                # Hyperparameter tuning
                best_model, best_cv_score, best_params = self.hyperparameter_tuning(
                    X_train, y_train, model_name, config
                )
                
                # Cross-validation
                cv_results = self.comprehensive_cross_validation(best_model, X_train, y_train, model_name)
                
                # Comprehensive evaluation
                evaluation_results = self.evaluate_model_performance(
                    best_model, X_sets, y_sets, model_name, feature_names
                )
                
                # Store results
                training_results[model_name] = {
                    'model': best_model,
                    'cv_results': cv_results,
                    'evaluation': evaluation_results,
                    'best_params': best_params,
                    'best_cv_score': best_cv_score
                }
                
                base_models[model_name] = best_model
                
                # Save model
                self.save_trained_model(best_model, model_name, target_name)
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
                continue
        
        # Create ensembles
        print(f"\n🔗 Creating ensemble models...")
        
        # Voting Ensemble
        voting_ensemble = self.create_voting_ensemble(base_models, X_train, y_train)
        voting_results = self.evaluate_model_performance(voting_ensemble, X_sets, y_sets, "Voting Ensemble", feature_names)
        training_results['Voting_Ensemble'] = {
            'model': voting_ensemble,
            'evaluation': voting_results
        }
        self.save_trained_model(voting_ensemble, 'voting_ensemble', target_name)
        
        # Stacking Ensemble
        stacking_ensemble = self.create_advanced_ensemble(base_models, X_train, y_train)
        stacking_results = self.evaluate_model_performance(stacking_ensemble, X_sets, y_sets, "Stacking Ensemble", feature_names)
        training_results['Stacking_Ensemble'] = {
            'model': stacking_ensemble,
            'evaluation': stacking_results
        }
        self.save_trained_model(stacking_ensemble, 'stacking_ensemble', target_name)
        
        # Determine best model
        best_model_name = None
        best_test_r2 = -np.inf
        
        for model_name, results in training_results.items():
            test_r2 = results['evaluation']['test']['metrics']['r2']
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_model_name = model_name
        
        self.best_models[target_name] = training_results[best_model_name]['model']
        
        print(f"\n🏆 Best model for {target_name}: {best_model_name} (Test R²: {best_test_r2:.4f})")
        
        return training_results

def run_complete_training(datasets):
    """Run complete training pipeline for all targets"""
    
    trainer = ProfessionalModelTrainer()
    all_results = {}
    
    for target_name in datasets.keys():
        results = trainer.train_models_for_target(datasets, target_name)
        all_results[target_name] = results
    
    return all_results, trainer
