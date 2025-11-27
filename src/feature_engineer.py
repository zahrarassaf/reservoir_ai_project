"""
PRODUCTION FEATURE ENGINEERING PIPELINE
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ReservoirFeatureEngineer:
    """INDUSTRY-GRADE FEATURE ENGINEERING"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """CREATE ADVANCED TEMPORAL FEATURES"""
        df = df.copy().sort_values(['well_id', 'timestamp'])
        
        for well_id in df['well_id'].unique():
            well_mask = df['well_id'] == well_id
            well_data = df[well_mask]
            
            # TEMPORAL DERIVATIVES
            df.loc[well_mask, 'pressure_derivative'] = well_data['bottomhole_pressure'].diff().fillna(0)
            df.loc[well_mask, 'oil_rate_derivative'] = well_data['oil_rate'].diff().fillna(0)
            
            # MOVING AVERAGES
            for window in [7, 30]:
                df.loc[well_mask, f'pressure_ma_{window}'] = well_data['bottomhole_pressure'].rolling(window, min_periods=1).mean()
                df.loc[well_mask, f'oil_rate_ma_{window}'] = well_data['oil_rate'].rolling(window, min_periods=1).mean()
            
            # CUMULATIVE FEATURES
            df.loc[well_mask, 'cumulative_oil'] = well_data['oil_rate'].cumsum()
            df.loc[well_mask, 'cumulative_water'] = well_data['water_rate'].cumsum()
        
        # WELL INTERACTION FEATURES
        df['well_density'] = df.groupby('time_index')['well_id'].transform('count')
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    def create_sequences(self, df: pd.DataFrame, target_col: str = 'oil_rate') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """CREATE TIME SEQUENCES FOR DEEP LEARNING"""
        df_engineered = self.create_temporal_features(df)
        
        # SELECT FEATURE COLUMNS
        exclude_cols = ['well_name', 'well_type', 'timestamp', 'years', 'time_index', 'data_source']
        feature_cols = [col for col in df_engineered.columns 
                       if col not in exclude_cols and col != target_col]
        
        self.feature_names = feature_cols
        
        sequences = []
        targets = []
        
        for well_id in df_engineered['well_id'].unique():
            well_data = df_engineered[df_engineered['well_id'] == well_id]
            well_data = well_data.sort_values('timestamp')
            
            values = well_data[feature_cols].values
            target_vals = well_data[target_col].values
            
            # CREATE SEQUENCES
            for i in range(len(values) - config.SEQUENCE_LENGTH):
                seq = values[i:(i + config.SEQUENCE_LENGTH)]
                target = target_vals[i + config.SEQUENCE_LENGTH]
                
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"✅ SEQUENCES CREATED: X{X.shape}, y{y.shape}")
        return X, y, feature_cols
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """COMPLETE FEATURE PREPARATION PIPELINE"""
        X, y, feature_names = self.create_sequences(df)
        
        # SCALE FEATURES
        if len(X) > 0:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
        
        return X, y, feature_names, df
