import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import glob
import logging

logger = logging.getLogger(__name__)


class ReservoirData:
    """Container for reservoir data"""
    
    def __init__(self):
        self.production = pd.DataFrame()
        self.pressure = np.array([])
        self.time = np.array([])
        self.wells = []
        self.metadata = {}
    
    def load_csv(self, filepath: str):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Try different CSV formats
            if 'Date' in df.columns and 'Oil' in df.columns:
                # Format 1: Date, Oil, Water, Gas columns
                df['Date'] = pd.to_datetime(df['Date'])
                self.time = np.arange(len(df))
                self.production = pd.DataFrame({
                    'Oil': df['Oil'].values,
                    'Water': df.get('Water', 0).values,
                    'Gas': df.get('Gas', 0).values
                })
                if 'Pressure' in df.columns:
                    self.pressure = df['Pressure'].values
                    
            elif 'TIME' in df.columns and 'RATE' in df.columns:
                # Format 2: TIME, RATE, PRESSURE columns
                self.time = df['TIME'].values
                self.production = pd.DataFrame({'Rate': df['RATE'].values})
                if 'PRESSURE' in df.columns:
                    self.pressure = df['PRESSURE'].values
                    
            elif len(df.columns) >= 2:
                # Format 3: Generic - assume first column is time, second is rate
                self.time = df.iloc[:, 0].values
                rate_col = df.iloc[:, 1].values
                self.production = pd.DataFrame({'Rate': rate_col})
                
                # Check for pressure column
                if len(df.columns) >= 3:
                    self.pressure = df.iloc[:, 2].values
                    
            else:
                raise ValueError("Unsupported CSV format")
            
            # Create well names
            self.wells = list(self.production.columns)
            
            logger.info(f"Loaded {len(self.time)} data points from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            return False
    
    def load_multiple_csv(self, directory: str, pattern: str = "*.csv"):
        """Load multiple CSV files from directory"""
        files = glob.glob(os.path.join(directory, pattern))
        
        if not files:
            logger.warning(f"No CSV files found in {directory}")
            return False
        
        all_data = []
        
        for file in files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not all_data:
            return False
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Try to identify columns
        if 'Date' in combined_df.columns:
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            self.time = np.arange(len(combined_df))
        else:
            self.time = np.arange(len(combined_df))
        
        # Identify production columns
        prod_columns = []
        for col in combined_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['oil', 'rate', 'prod', 'bbl', 'stb']):
                prod_columns.append(col)
        
        if prod_columns:
            self.production = combined_df[prod_columns]
        else:
            # Use first numeric column as production
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.production = pd.DataFrame({numeric_cols[0]: combined_df[numeric_cols[0]].values})
        
        # Identify pressure column
        pressure_cols = [col for col in combined_df.columns if 'pressure' in col.lower()]
        if pressure_cols:
            self.pressure = combined_df[pressure_cols[0]].values
        elif 'Pressure' in combined_df.columns:
            self.pressure = combined_df['Pressure'].values
        
        self.wells = list(self.production.columns)
        
        logger.info(f"Loaded {len(self.time)} data points from {len(files)} files")
        return True
    
    def create_sample_data(self, n_days: int = 1825, n_wells: int = 6):
        """Create sample data for testing"""
        np.random.seed(42)
        
        # Time array
        self.time = np.arange(n_days)
        
        # Create production data for multiple wells
        production_data = {}
        
        for i in range(n_wells):
            # Exponential decline with noise
            qi = 1000 + np.random.uniform(-200, 200)
            Di = 0.001 + np.random.uniform(-0.0002, 0.0002)
            
            rates = qi * np.exp(-Di * self.time)
            noise = np.random.normal(0, 50, n_days)
            rates = np.maximum(0, rates + noise)
            
            production_data[f'Well_{i+1}'] = rates
        
        self.production = pd.DataFrame(production_data)
        
        # Create pressure data
        initial_pressure = 4000
        pressure_decline = 0.0001 * self.time
        noise = np.random.normal(0, 50, n_days)
        self.pressure = np.maximum(1000, initial_pressure - pressure_decline + noise)
        
        self.wells = list(self.production.columns)
        
        logger.info(f"Created sample data: {n_days} days, {n_wells} wells")
        return True
    
    @property
    def has_production_data(self) -> bool:
        """Check if production data exists"""
        return not self.production.empty and len(self.production) > 0
    
    @property
    def has_pressure_data(self) -> bool:
        """Check if pressure data exists"""
        return len(self.pressure) > 0
    
    def summary(self) -> Dict[str, Any]:
        """Get data summary"""
        return {
            'wells': len(self.wells),
            'time_points': len(self.time),
            'production_columns': list(self.production.columns),
            'pressure_available': self.has_pressure_data,
            'production_range': {
                'min': float(self.production.min().min()) if self.has_production_data else 0,
                'max': float(self.production.max().max()) if self.has_production_data else 0,
                'mean': float(self.production.mean().mean()) if self.has_production_data else 0
            },
            'pressure_range': {
                'min': float(self.pressure.min()) if self.has_pressure_data else 0,
                'max': float(self.pressure.max()) if self.has_pressure_data else 0,
                'mean': float(self.pressure.mean()) if self.has_pressure_data else 0
            }
        }
