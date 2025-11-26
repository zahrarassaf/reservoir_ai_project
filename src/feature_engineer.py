"""
INDUSTRY-GRADE FEATURE ENGINEERING FOR RESERVOIR FORECASTING
PRODUCTION-READY FEATURE PIPELINE
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

from .config import config

class ReservoirFeatureEngineer:
    """PRODUCTION-GRADE FEATURE ENGINEERING FOR RESERVOIR DATA"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """CREATE ADVANCED RESERVOIR ENGINEERING FEATURES"""
        print("🛠️ CREATING ADVANCED RESERVOIR FEATURES...")
        
        df = df.copy()
        
        # TEMPORAL FEATURES
        df['TIME_SIN'] = np.sin(2 * np.pi * df['YEARS'] / 10)
        df['TIME_COS'] = np.cos(2 * np.pi * df['YEARS'] / 10)
        
        # RATE OF CHANGE FEATURES
        df.sort_values(['WELL_ID', 'TIME_INDEX'], inplace=True)
        
        for well_id in df['WELL_ID'].unique():
            well_mask = df['WELL_ID'] == well_id
            well_data = df[well_mask]
            
            # RATE DERIVATIVES
            df.loc[well_mask, 'DP_DT'] = well_data['BOTTOMHOLE_PRESSURE'].diff() / well_data['TIME_INDEX'].diff()
            df.loc[well_mask, 'DQ_OIL_DT'] = well_data['FLOW_RATE_OIL'].diff() / well_data['TIME_INDEX'].diff()
            df.loc[well_mask, 'DQ_WATER_DT'] = well_data['FLOW_RATE_WATER'].diff() / well_data['TIME_INDEX'].diff()
            
            # MOVING AVERAGES
            df.loc[well_mask, 'PRESSURE_MA_5'] = well_data['BOTTOMHOLE_PRESSURE'].rolling(5).mean()
            df.loc[well_mask, 'OIL_RATE_MA_5'] = well_data['FLOW_RATE_OIL'].rolling(5).mean()
            df.loc[well_mask, 'WATER_RATE_MA_5'] = well_data['FLOW_RATE_WATER'].rolling(5).mean()
            
            # CUMULATIVE RATIOS
            df.loc[well_mask, 'WCUT'] = well_data['FLOW_RATE_WATER'] / (well_data['FLOW_RATE_OIL'] + well_data['FLOW_RATE_WATER'] + 1e-8)
            df.loc[well_mask, 'GOR'] = well_data['FLOW_RATE_GAS'] / (well_data['FLOW_RATE_OIL'] + 1e-8)
            
            # RECOVERY FACTOR ESTIMATES
            df.loc[well_mask, 'RF_OIL'] = well_data['CUMULATIVE_OIL'] / (well_data['CUMULATIVE_OIL'].max() + 1e-8)
            df.loc[well_mask, 'RF_WATER'] = well_data['CUMULATIVE_WATER'] / (well_data['CUMULATIVE_WATER'].max() + 1e-8)
        
        # WELL INTERFERENCE FEATURES (SIMPLIFIED)
        df['WELL_DENSITY'] = df.groupby('TIME_INDEX')['WELL_ID'].transform('count')
        
        # FILL NaN VALUES FROM ROLLING CALCULATIONS
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        print(f"✅ ADVANCED FEATURES CREATED: {df.shape}")
        return df
    
    def create_sequences(self, df: pd.DataFrame, target_column: str = 'FLOW_RATE_OIL') -> tuple:
        """CREATE TIME SEQUENCES FOR DEEP LEARNING MODELS"""
        print("🔄 CREATING TIME SEQUENCES...")
        
        sequences = []
        targets = []
        well_groups = df.groupby('WELL_ID')
        
        feature_columns = [col for col in df.columns if col not in [
            'DATASET', 'WELL_NAME', 'WELL_TYPE', 'WELL_ID', 'TIME_INDEX', 'YEARS'
        ]]
        
        for well_id, well_data in well_groups:
            well_data = well_data.sort_values('TIME_INDEX')
            values = well_data[feature_columns].values
            
            for i in range(len(values) - config.SEQUENCE_LENGTH):
                seq = values[i:(i + config.SEQUENCE_LENGTH)]
                target = values[i + config.SEQUENCE_LENGTH][feature_columns.index(target_column)]
                
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"✅ SEQUENCES CREATED: X={X.shape}, y={y.shape}")
        return X, y, feature_columns
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """COMPLETE FEATURE PREPARATION PIPELINE"""
        # CREATE ADVANCED FEATURES
        df_engineered = self.create_advanced_features(df)
        
        # CREATE SEQUENCES
        X, y, feature_names = self.create_sequences(df_engineered)
        
        return X, y, feature_names, df_engineered
