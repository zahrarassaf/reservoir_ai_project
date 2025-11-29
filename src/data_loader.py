import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class ReservoirDataLoader:
    def __init__(self, config):
        self.config = config
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        
    def load_and_validate_data(self, data_path=None):
        """Load and rigorously validate reservoir data"""
        try:
            if data_path:
                df = pd.read_csv(data_path)
            else:
                df = self._generate_synthetic_data()
            
            print("🔍 DATA VALIDATION REPORT")
            print("=========================")
            print(f"📊 Dataset shape: {df.shape}")
            print(f"🎯 Target variable: oil_rate")
            
            # Data quality checks
            self._perform_data_quality_checks(df)
            
            # Validate temporal structure
            self._validate_temporal_structure(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            raise
    
    def _generate_synthetic_data(self):
        """Generate realistic synthetic reservoir data for testing"""
        np.random.seed(self.config.RANDOM_STATE)
        n_samples = 24000
        n_wells = 24
        time_steps = 1000
        
        data = []
        for well in range(n_wells):
            base_pressure = np.random.uniform(2000, 5000)
            base_rate = np.random.uniform(100, 2000)
            
            for t in range(time_steps):
                # Realistic reservoir features
                pressure = base_pressure - t * 0.1 + np.random.normal(0, 50)
                water_cut = np.clip(0.1 + t * 0.0001 + np.random.normal(0, 0.02), 0, 0.9)
                gas_oil_ratio = np.random.lognormal(6, 0.5)
                
                # Realistic oil rate (target)
                trend = base_rate * np.exp(-t * 0.0005)
                seasonal = 100 * np.sin(2 * np.pi * t / 365)
                noise = np.random.normal(0, 50)
                oil_rate = np.clip(trend + seasonal + noise, 0, None)
                
                row = {
                    'well_id': f'well_{well:02d}',
                    'time_step': t,
                    'pressure': pressure,
                    'water_cut': water_cut,
                    'gas_oil_ratio': gas_oil_ratio,
                    'injection_rate': np.random.uniform(0, 1000),
                    'bottomhole_flowing_pressure': pressure - np.random.uniform(100, 500),
                    'wellhead_pressure': pressure - np.random.uniform(500, 1000),
                    'temperature': np.random.uniform(80, 120),
                    'choke_size': np.random.uniform(10, 100),
                    'gas_lift_rate': np.random.uniform(0, 500),
                    'water_injection_pressure': np.random.uniform(1000, 3000),
                    'productivity_index': oil_rate / max(1, pressure - 1500),
                    'reservoir_thickness': np.random.uniform(50, 200),
                    'permeability': np.random.lognormal(3, 0.5),
                    'oil_rate': oil_rate
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def _perform_data_quality_checks(self, df):
        """Comprehensive data quality assessment"""
        print("\n📋 DATA QUALITY CHECKS:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {df.duplicated().sum()}")
        print(f"   Zero variance features: {len(df.columns[df.nunique() == 1])}")
        
        # Statistical summary
        print(f"   Target stats - Mean: {df['oil_rate'].mean():.2f}, Std: {df['oil_rate'].std():.2f}")
        print(f"   Skewness: {df['oil_rate'].skew():.2f}")
    
    def _validate_temporal_structure(self, df):
        """Validate time series structure"""
        if 'time_step' in df.columns:
            time_gaps = df['time_step'].diff().value_counts()
            print(f"   Temporal gaps: {dict(time_gaps.head())}")
    
    def create_sequences(self, df, target_col='oil_rate'):
        """Create time series sequences with proper validation"""
        feature_cols = [col for col in df.columns if col not in ['well_id', 'time_step', target_col]]
        
        print(f"\n🎯 SEQUENCE CREATION:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Sequence length: {self.config.SEQUENCE_LENGTH}")
        
        X, y = [], []
        
        # Group by well for realistic time series
        for well_id, well_data in df.groupby('well_id'):
            well_data = well_data.sort_values('time_step')
            features = well_data[feature_cols].values
            targets = well_data[target_col].values
            
            for i in range(len(well_data) - self.config.SEQUENCE_LENGTH):
                X.append(features[i:(i + self.config.SEQUENCE_LENGTH)])
                y.append(targets[i + self.config.SEQUENCE_LENGTH])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"   Final sequences: X{X.shape}, y{y.shape}")
        return X, y, feature_cols
    
    def scale_features(self, X_train, X_test, y_train, y_test):
        """Robust feature scaling"""
        # Reshape for scaling
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train_flat)
        X_test_scaled = self.feature_scaler.transform(X_test_flat)
        
        # Scale target
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def inverse_scale_target(self, y_scaled):
        """Inverse transform target variable"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
