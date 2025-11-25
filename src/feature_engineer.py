"""
Advanced feature engineering for reservoir forecasting
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

from .config import config

class FeatureEngineer:
    """Creates temporal, spatial, and domain-specific features"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling window features"""
        df_sorted = df.sort_values(['Well', 'Time']).copy()
        
        feature_dfs = []
        
        for well in df_sorted['Well'].unique():
            well_data = df_sorted[df_sorted['Well'] == well].copy()
            
            # Lag features for key dynamic variables
            for lag in config.LAGS:
                well_data[f'Pressure_lag_{lag}'] = well_data['Pressure'].shift(lag)
                well_data[f'FlowRate_lag_{lag}'] = well_data['FlowRate'].shift(lag)
                well_data[f'Saturation_lag_{lag}'] = well_data['Saturation'].shift(lag)
            
            # Rolling statistics
            for window in config.ROLLING_WINDOWS:
                # Pressure rolling features
                well_data[f'Pressure_roll_mean_{window}'] = (
                    well_data['Pressure'].rolling(window, min_periods=1).mean()
                )
                well_data[f'Pressure_roll_std_{window}'] = (
                    well_data['Pressure'].rolling(window, min_periods=1).std()
                )
                
                # Flow rate rolling features
                well_data[f'FlowRate_roll_mean_{window}'] = (
                    well_data['FlowRate'].rolling(window, min_periods=1).mean()
                )
                well_data[f'FlowRate_roll_std_{window}'] = (
                    well_data['FlowRate'].rolling(window, min_periods=1).std()
                )
            
            # Rate of change features
            well_data['Pressure_roc'] = well_data['Pressure'].pct_change()
            well_data['FlowRate_roc'] = well_data['FlowRate'].pct_change()
            
            # Cumulative features
            well_data['Cumulative_Flow'] = well_data['FlowRate'].cumsum()
            
            feature_dfs.append(well_data)
        
        features_df = pd.concat(feature_dfs).reset_index(drop=True)
        
        # Handle infinite values from percentage changes
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features_df
    
    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific engineering features"""
        df_features = df.copy()
        
        # Productivity index approximation
        df_features['Productivity_Index'] = df_features['FlowRate'] / (
            df_features['Pressure'] + 1e-8
        )
        
        # Mobility ratio approximation
        df_features['Mobility_Ratio'] = (
            df_features['Permeability'] / (df_features['Saturation'] + 1e-8)
        )
        
        # Reservoir energy indicator
        df_features['Energy_Indicator'] = (
            df_features['Pressure'] * df_features['Porosity'] * 
            df_features['ReservoirQuality']
        )
        
        # Well interaction features (simplified)
        df_features['Well_Density'] = df_features.groupby('Time')['Well'].transform('count')
        
        return df_features
    
    def prepare_sequences(self, features_df: pd.DataFrame, target_col: str = 'FlowRate',
                         sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert tabular data to sequences for temporal models"""
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
        
        sequences = []
        targets = []
        
        # Get feature columns (exclude identifiers and target)
        exclude_cols = ['Time', 'Well', 'WellType', target_col]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        for well in features_df['Well'].unique():
            well_data = features_df[features_df['Well'] == well]
            well_features = well_data[feature_cols].values
            well_target = well_data[target_col].values
            
            # Create sequences for this well
            for i in range(len(well_data) - sequence_length):
                sequences.append(well_features[i:i + sequence_length])
                targets.append(well_target[i + sequence_length])
        
        self.feature_names = feature_cols
        
        return np.array(sequences), np.array(targets)
    
    def prepare_tabular_data(self, features_df: pd.DataFrame, 
                           target_col: str = 'FlowRate') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for traditional ML models"""
        # Get feature columns
        exclude_cols = ['Time', 'Well', 'WellType', target_col]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """Scale features for ML models"""
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['features'] = scaler
        
        return X_train_scaled, X_test_scaled
