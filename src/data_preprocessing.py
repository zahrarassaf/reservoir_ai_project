"""
Professional Data Preprocessing Module
Handles data cleaning, validation, and preparation with industry best practices
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import config

class DataPreprocessor:
    """
    A professional data preprocessing class for well log data
    Implements industry standards for petrophysical data preparation
    """
    
    def __init__(self, use_robust_scaling=True):
        self.use_robust_scaling = use_robust_scaling
        self.scalers = {}
        self.imputers = {}
        self.feature_names = config.ORIGINAL_FEATURES
        self.data_quality_report = {}
        
    def generate_data_quality_report(self, df):
        """Comprehensive data quality assessment"""
        report = {
            'original_shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'statistical_summary': df.describe().to_dict()
        }
        
        # Data quality scores
        quality_metrics = {
            'completeness_score': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplication_score': (1 - report['duplicate_rows'] / df.shape[0]) * 100
        }
        
        report['quality_metrics'] = quality_metrics
        self.data_quality_report = report
        
        print("📊 Data Quality Report:")
        print(f"   • Completeness: {quality_metrics['completeness_score']:.1f}%")
        print(f"   • Duplication: {quality_metrics['duplication_score']:.1f}%")
        print(f"   • Missing values: {sum(report['missing_values'].values())}")
        
        return report
    
    def load_and_validate_data(self):
        """Load data with comprehensive validation"""
        try:
            df = pd.read_csv(config.DATA_FILE)
            print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Data quality assessment
            self.generate_data_quality_report(df)
            
            # Validate required columns
            missing_columns = set(config.ORIGINAL_FEATURES + config.TARGETS) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Data file not found: {config.DATA_FILE}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df, strategy='median'):
        """Advanced missing value handling"""
        df_clean = df.copy()
        
        # Separate features and targets
        features = [col for col in config.ORIGINAL_FEATURES if col in df_clean.columns]
        targets = [col for col in config.TARGETS if col in df_clean.columns]
        
        # Impute features
        for feature in features:
            if df_clean[feature].isnull().any():
                if strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                
                df_clean[feature] = imputer.fit_transform(df_clean[[feature]]).ravel()
                self.imputers[feature] = imputer
        
        # For targets, remove rows with missing values
        initial_size = len(df_clean)
        df_clean = df_clean.dropna(subset=targets)
        removed_count = initial_size - len(df_clean)
        
        if removed_count > 0:
            print(f"⚠️  Removed {removed_count} rows with missing target values")
        
        return df_clean
    
    def detect_and_handle_outliers(self, df, method='IQR'):
        """Outlier detection and handling for well log data"""
        df_clean = df.copy()
        outlier_report = {}
        
        for feature in config.ORIGINAL_FEATURES:
            if feature in df_clean.columns:
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_clean[(df_clean[feature] < lower_bound) | (df_clean[feature] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    # Cap outliers instead of removing (common in petrophysical data)
                    df_clean[feature] = np.where(df_clean[feature] < lower_bound, lower_bound, df_clean[feature])
                    df_clean[feature] = np.where(df_clean[feature] > upper_bound, upper_bound, df_clean[feature])
                    
                    outlier_report[feature] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(df_clean)) * 100,
                        'method': 'capping'
                    }
        
        if outlier_report:
            print("📊 Outlier Handling Report:")
            for feature, report in outlier_report.items():
                print(f"   • {feature}: {report['count']} outliers ({report['percentage']:.1f}%) - {report['method']}")
        
        return df_clean, outlier_report
    
    def create_validation_set(self, X, y):
        """Create train/validation/test split"""
        # First split: train + validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=pd.cut(y, bins=5) if len(y) > 100 else None
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        print(f"📊 Dataset Split:")
        print(f"   • Train: {X_train.shape[0]} samples")
        print(f"   • Validation: {X_val.shape[0]} samples") 
        print(f"   • Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, feature_set_name):
        """Scale features with proper data leakage prevention"""
        if self.use_robust_scaling:
            scaler = RobustScaler()  # Better for outliers
        else:
            scaler = StandardScaler()
        
        # Fit only on training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        self.scalers[feature_set_name] = scaler
        joblib.dump(scaler, config.MODELS_DIR / f'scaler_{feature_set_name}.pkl')
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def prepare_datasets(self):
        """Main method to prepare all datasets"""
        print("🚀 Starting professional data preprocessing pipeline...")
        
        # Load and validate data
        df = self.load_and_validate_data()
        if df is None:
            return None
        
        # Handle missing values
        df_clean = self.handle_missing_values(df, strategy='median')
        
        # Handle outliers
        df_final, outlier_report = self.detect_and_handle_outliers(df_clean)
        
        # Prepare features and targets
        feature_columns = [col for col in config.ORIGINAL_FEATURES if col in df_final.columns]
        X = df_final[feature_columns]
        
        datasets = {}
        
        for target in config.TARGETS:
            if target in df_final.columns:
                print(f"\n🎯 Preparing datasets for: {target}")
                y = df_final[target]
                
                # Create validation split
                X_train, X_val, X_test, y_train, y_val, y_test = self.create_validation_set(X, y)
                
                # Scale features
                X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
                    X_train, X_val, X_test, target
                )
                
                datasets[target] = {
                    'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                    'feature_names': feature_columns,
                    'scaler': self.scalers[target]
                }
        
        print("✅ Data preprocessing completed successfully!")
        return datasets

# Usage example
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    datasets = preprocessor.prepare_datasets()
