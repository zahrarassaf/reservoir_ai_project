"""
Advanced Feature Engineering for Petrophysical Data
Domain-specific feature creation and selection
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import config

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for well log data
    Implements domain-specific transformations and feature creation
    """
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.kmeans = None
        self.engineered_features = []
        
    def create_domain_specific_features(self, df, feature_columns):
        """Create domain-specific features for petrophysical analysis"""
        df_engineered = df.copy()
        
        # 1. Geophysical Ratios (common in petrophysics)
        if all(col in df_columns for col in ['GR', 'PHIND']):
            df_engineered['GR_PHIND_ratio'] = df_engineered['GR'] / (df_engineered['PHIND'] + 1e-8)
            self.engineered_features.append('GR_PHIND_ratio')
        
        if all(col in df_columns for col in ['ILD_log10', 'PHIND']):
            df_engineered['Resistivity_Porosity_Index'] = df_engineered['ILD_log10'] * df_engineered['PHIND']
            self.engineered_features.append('Resistivity_Porosity_Index')
        
        # 2. Log-derived properties
        if 'PHIND' in df_columns:
            df_engineered['PHIND_squared'] = df_engineered['PHIND'] ** 2
            df_engineered['PHIND_log'] = np.log1p(np.abs(df_engineered['PHIND']))
            self.engineered_features.extend(['PHIND_squared', 'PHIND_log'])
        
        # 3. Statistical features using rolling windows
        for column in feature_columns:
            if column in ['GR', 'ILD_log10', 'PHIND']:
                # Rolling statistics (capture local trends)
                df_engineered[f'{column}_rolling_mean_5'] = df_engineered[column].rolling(5, min_periods=1).mean()
                df_engineered[f'{column}_rolling_std_5'] = df_engineered[column].rolling(5, min_periods=1).std()
                
                # Difference features
                df_engineered[f'{column}_diff'] = df_engineered[column].diff().fillna(0)
                
                self.engineered_features.extend([
                    f'{column}_rolling_mean_5', 
                    f'{column}_rolling_std_5',
                    f'{column}_diff'
                ])
        
        # 4. Interaction terms
        if all(col in df_columns for col in ['GR', 'PE']):
            df_engineered['GR_PE_interaction'] = df_engineered['GR'] * df_engineered['PE']
            self.engineered_features.append('GR_PE_interaction')
        
        print(f"✅ Created {len(self.engineered_features)} engineered features")
        return df_engineered
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        X_pca = self.pca.fit_transform(X)
        
        print(f"✅ PCA reduced features from {X.shape[1]} to {X_pca.shape[1]} components")
        print(f"   Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca
    
    def cluster_analysis(self, X, n_clusters=3):
        """Apply K-means clustering for segment identification"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE)
        clusters = self.kmeans.fit_predict(X)
        
        # Add cluster features
        cluster_dummies = pd.get_dummies(clusters, prefix='cluster')
        
        print(f"✅ Created {n_clusters} clusters for segment analysis")
        return cluster_dummies
    
    def select_features_rfe(self, X, y, estimator, n_features=10):
        """Recursive Feature Elimination for feature selection"""
        self.feature_selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=1
        )
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_mask = self.feature_selector.support_
        
        print(f"✅ RFE selected {np.sum(selected_mask)} features from {X.shape[1]} total")
        return X_selected, selected_mask
    
    def select_features_statistical(self, X, y, k=10):
        """Statistical feature selection using mutual information"""
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        
        feature_scores = dict(zip(range(X.shape[1]), selector.scores_))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("📊 Top features by mutual information:")
        for idx, (feature_idx, score) in enumerate(sorted_features[:5]):
            print(f"   {idx+1}. Feature {feature_idx}: {score:.4f}")
        
        return X_selected, selected_mask
    
    def create_comprehensive_features(self, X, feature_names, y=None):
        """Comprehensive feature engineering pipeline"""
        print("🔧 Starting advanced feature engineering...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        
        # 1. Domain-specific features
        df_engineered = self.create_domain_specific_features(df, feature_names)
        
        # 2. Statistical feature selection if target is provided
        if y is not None:
            X_combined = df_engineered.values
            X_selected, selected_mask = self.select_features_statistical(X_combined, y, k=15)
            
            selected_features = [
                feature for feature, selected in 
                zip(df_engineered.columns, selected_mask) if selected
            ]
            
            print(f"✅ Final feature set: {len(selected_features)} features")
            return X_selected, selected_features
        
        return df_engineered.values, df_engineered.columns.tolist()

def engineer_features_for_target(datasets, target_name):
    """Apply feature engineering for specific target"""
    from sklearn.ensemble import RandomForestRegressor
    
    engineer = AdvancedFeatureEngineer()
    
    # Get training data
    X_train = datasets[target_name]['X_train']
    y_train = datasets[target_name]['y_train']
    feature_names = datasets[target_name]['feature_names']
    
    # Apply feature engineering
    X_engineered, new_feature_names = engineer.create_comprehensive_features(
        X_train, feature_names, y_train
    )
    
    # Update datasets with engineered features
    datasets[target_name]['X_train'] = X_engineered
    datasets[target_name]['feature_names'] = new_feature_names
    
    # Also transform validation and test sets
    if 'X_val' in datasets[target_name]:
        X_val_engineered, _ = engineer.create_comprehensive_features(
            datasets[target_name]['X_val'], feature_names
        )
        datasets[target_name]['X_val'] = X_val_engineered
    
    if 'X_test' in datasets[target_name]:
        X_test_engineered, _ = engineer.create_comprehensive_features(
            datasets[target_name]['X_test'], feature_names
        )
        datasets[target_name]['X_test'] = X_test_engineered
    
    return datasets, engineer
