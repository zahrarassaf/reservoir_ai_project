"""
DATA PREPROCESSING MODULE
QUICK FIX FOR MISSING IMPORTS
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Quick data preprocessing for reservoir data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df):
        """Basic preprocessing pipeline"""
        # Select numerical features
        numerical_features = ['BOTTOMHOLE_PRESSURE', 'FLOW_RATE_OIL', 
                             'FLOW_RATE_WATER', 'FLOW_RATE_GAS']
        
        # Handle missing values
        df_clean = df[numerical_features].fillna(method='ffill')
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df_clean)
        
        return scaled_data, numerical_features
