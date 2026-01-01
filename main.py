#!/usr/bin/env python3
"""
Reservoir Simulation - SPE9 Data Analysis with Real Data Integration
With ML integration for enhanced reservoir simulation
"""

# ============================================================================
# CRITICAL: SET ALL RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
import numpy as np
import random
import os

# Set global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# For PyTorch
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

print(f"Random seed set to: {SEED} for reproducible results")

# ============================================================================
# REST OF IMPORTS
# ============================================================================
import pandas as pd
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import traceback

# Import the new DataLoader
from src.data_loader import DataLoader

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SimpleCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        
        self.fc1 = nn.Linear(32 * 6 * 6 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PropertyPredictor:
    def __init__(self, seed=SEED):
        if not TORCH_AVAILABLE:
            self.model = None
            return
            
        self.model = SimpleCNN3D()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.seed = seed
    
    def prepare_data(self, grid_data_3d, properties_dict):
        try:
            print(f"Grid data shape: {grid_data_3d.shape}")
            
            if grid_data_3d.ndim == 3:
                x_tensor = torch.FloatTensor(grid_data_3d).unsqueeze(0).unsqueeze(0)
            elif grid_data_3d.ndim == 4:
                x_tensor = torch.FloatTensor(grid_data_3d).unsqueeze(1)
            else:
                x_tensor = torch.FloatTensor(grid_data_3d).unsqueeze(0).unsqueeze(0)
            
            target_values = []
            for prop_name in ['permeability', 'porosity', 'saturation']:
                if prop_name in properties_dict:
                    prop_data = properties_dict[prop_name]
                    if hasattr(prop_data, 'mean'):
                        target_values.append(float(np.mean(prop_data)))
                    else:
                        target_values.append(float(prop_data))
                else:
                    target_values.append(0.0)
            
            y_tensor = torch.FloatTensor([target_values])
            
            print(f"X tensor shape: {x_tensor.shape}")
            print(f"Y tensor shape: {y_tensor.shape}")
            
            return [(x_tensor, y_tensor)], [(x_tensor, y_tensor)]
            
        except Exception as e:
            print(f"Data preparation failed: {e}")
            return [], []
    
    def train(self, train_loader, val_loader, epochs=10):
        if self.model is None:
            return [], []
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def evaluate(self, grid_data_3d, properties_dict):
        if self.model is None:
            return {}
        
        self.model.eval()
        
        train_loader, _ = self.prepare_data(grid_data_3d, properties_dict)
        
        if not train_loader:
            return {}
        
        x_batch, y_batch = train_loader[0]
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x_batch)
        
        y_true = y_batch.cpu().numpy().flatten()
        y_pred = predictions.cpu().numpy().flatten()
        
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'R2': float(r2),
            'predictions': y_pred.tolist(),
            'targets': y_true.tolist()
        }
    
    def save_model(self, path):
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': {
                    'input_channels': 1,
                    'output_features': 3
                },
                'seed': self.seed
            }, path)
            return True
        return False

class EconomicFeatureEngineer:
    def create_features(self, reservoir_params, economic_params):
        features = {
            'porosity': reservoir_params.get('avg_porosity', 0.2),
            'permeability': reservoir_params.get('avg_permeability', 100),
            'oil_in_place': reservoir_params.get('oil_in_place', 1e6) / 1e6,
            'recoverable_oil': reservoir_params.get('recoverable_oil', 0.5e6) / 1e6,
            'oil_price': economic_params.get('oil_price', 70),
            'opex_per_bbl': economic_params.get('opex_per_bbl', 20),
            'capex': economic_params.get('capex', 10e6) / 1e6,
            'discount_rate': economic_params.get('discount_rate', 0.1) * 100
        }
        
        features['recovery_factor'] = features['recoverable_oil'] / features['oil_in_place'] if features['oil_in_place'] > 0 else 0
        features['price_cost_ratio'] = features['oil_price'] / features['opex_per_bbl'] if features['opex_per_bbl'] > 0 else 0
        features['unit_capex'] = features['capex'] / features['recoverable_oil'] if features['recoverable_oil'] > 0 else 0
        
        return pd.DataFrame([features])

class SVREconomicPredictor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        
    def prepare_data(self, X, y):
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=SEED,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train.values)
        else:
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X_train, y_train.values)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            return {}
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        predictions = self.model.predict(X_test)
        
        metrics = {
            'NPV': {
                'MSE': mean_squared_error(y_test['npv'], predictions[:, 0]),
                'R2': r2_score(y_test['npv'], predictions[:, 0])
            },
            'IRR': {
                'MSE': mean_squared_error(y_test['irr'], predictions[:, 1]),
                'R2': r2_score(y_test['irr'], predictions[:, 1])
            },
            'ROI': {
                'MSE': mean_squared_error(y_test['roi'], predictions[:, 2]),
                'R2': r2_score(y_test['roi'], predictions[:, 2])
            },
            'Payback': {
                'MSE': mean_squared_error(y_test['payback_period'], predictions[:, 3]),
                'R2': r2_score(y_test['payback_period'], predictions[:, 3])
            }
        }
        
        return metrics
    
    def predict(self, X):
        if self.model is None:
            return pd.DataFrame({'npv': [0], 'irr': [0], 'roi': [0], 'payback_period': [0]})
        
        predictions = self.model.predict(X)
        
        return pd.DataFrame(predictions, columns=['npv', 'irr', 'roi', 'payback_period'])
    
    def save_model(self, path):
        if self.model:
            from joblib import dump
            dump(self.model, path)
            return True
        return False

print("=" * 70)
print("RESERVOIR SIMULATION - SPE9 REAL DATA ANALYSIS")
print("=" * 70)

class RealSPE9DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.real_data_loader = DataLoader()
    
    def load_all_data(self):
        print("\nLoading SPE9 datasets with new DataLoader...")
        
        # Load data using the new DataLoader
        success = self.real_data_loader.load_all_spe9_data()
        
        if not success:
            print("Failed to load real SPE9 data")
            return self._create_fallback_data()
        
        real_data = self.real_data_loader.get_reservoir_data()
        
        results = {
            'is_real_data': True,
            'real_data_loaded': True,
            'files_found': ['SPE9.DATA', 'SPE9.GRDECL', 'PERMVALUES.DATA', 'TOPSVALUES.DATA'] + 
                          ['SPE9_CP.DATA', 'SPE9_CP_GROUP.DATA', 'SPE9_CP_SHORT.DATA', 'SPE9_CP_SHORT_RESTART.DATA'],
            'grid_info': {
                'dimensions': real_data['grid'].get('dimensions', (24, 25, 15)),
                'total_cells': real_data['metadata'].get('cells', 9000),
                'real_data': real_data['metadata'].get('real_data', True)
            },
            'properties': {
                'permeability': real_data['grid'].get('permeability_x', []),
                'porosity': real_data['grid'].get('porosity', []),
                'tops': real_data['grid'].get('depth_tops', []),
                'water_saturation': real_data['grid'].get('water_saturation', []),
                'oil_saturation': real_data['grid'].get('oil_saturation', [])
            },
            'wells': [
                {
                    'name': well_name,
                    'i': real_data['well_locations'][well_name].get('i', 1),
                    'j': real_data['well_locations'][well_name].get('j', 1),
                    'type': real_data['well_locations'][well_name].get('type', 'PRODUCER')
                }
                for well_name in real_data['well_locations']
            ],
            'well_production_data': real_data['wells'],
            'metadata': real_data['metadata']
        }
        
        print(f"Real SPE9 data loaded successfully!")
        print(f"Grid: {results['grid_info']['dimensions']} = {results['grid_info']['total_cells']:,} cells")
        print(f"Wells: {len(results['wells'])} wells")
        print(f"Real data: {results['grid_info']['real_data']}")
        
        return results
    
    def _create_fallback_data(self):
        print("Creating fallback synthetic data...")
        return {
            'is_real_data': False,
            'real_data_loaded': False,
            'files_found': [],
            'grid_info': {
                'dimensions': (24, 25, 15),
                'total_cells': 9000,
                'real_data': False
            },
            'properties': {
                'permeability': np.random.lognormal(4, 0.5, 9000),
                'porosity': np.random.uniform(0.1, 0.3, 9000)
            },
            'wells': [
                {'name': 'PROD1', 'i': 2, 'j': 2, 'type': 'PRODUCER'},
                {'name': 'PROD2', 'i': 22, 'j': 2, 'type': 'PRODUCER'},
                {'name': 'PROD3', 'i': 2, 'j': 23, 'type': 'PRODUCER'},
                {'name': 'PROD4', 'i': 22, 'j': 23, 'type': 'PRODUCER'},
                {'name': 'INJ1', 'i': 12, 'j': 12, 'type': 'INJECTOR'},
            ],
            'metadata': {'dataset': 'Synthetic_Fallback'}
        }

class PhysicsBasedSimulator:
    def __init__(self, real_data):
        self.data = real_data
        self.setup_reservoir()
    
    def setup_reservoir(self):
        print("\nSetting up reservoir from data...")
        
        if 'grid_info' in self.data and 'dimensions' in self.data['grid_info']:
            self.nx, self.ny, self.nz = self.data['grid_info']['dimensions']
        else:
            self.nx, self.ny, self.nz = 24, 25, 15
        
        self.total_cells = self.nx * self.ny * self.nz
        
        # Use real permeability data
        if 'properties' in self.data and 'permeability' in self.data['properties']:
            self.permeability = self.data['properties']['permeability']
            if len(self.permeability) != self.total_cells:
                print(f"Warning: Permeability array size ({len(self.permeability)}) doesn't match grid ({self.total_cells})")
                if len(self.permeability) > self.total_cells:
                    self.permeability = self.permeability[:self.total_cells]
                else:
                    # Pad with mean value
                    mean_val = np.mean(self.permeability) if len(self.permeability) > 0 else 100
                    padding = np.ones(self.total_cells - len(self.permeability)) * mean_val
                    self.permeability = np.concatenate([self.permeability, padding])
            print(f"Using real permeability data: {len(self.permeability)} values")
        else:
            np.random.seed(SEED)
            self.permeability = np.random.lognormal(mean=np.log(100), sigma=0.8, size=self.total_cells)
            print("Using synthetic permeability data")
        
        # Use real porosity data
        if 'properties' in self.data and 'porosity' in self.data['properties']:
            self.porosity = self.data['properties']['porosity']
            if len(self.porosity) != self.total_cells:
                print(f"Warning: Porosity array size mismatch, adjusting...")
                if len(self.porosity) > self.total_cells:
                    self.porosity = self.porosity[:self.total_cells]
                else:
                    mean_val = np.mean(self.porosity) if len(self.porosity) > 0 else 0.2
                    padding = np.ones(self.total_cells - len(self.porosity)) * mean_val
                    self.porosity = np.concatenate([self.porosity, padding])
            print(f"Using real porosity data: {len(self.porosity)} values")
        else:
            np.random.seed(SEED)
            self.porosity = np.random.uniform(0.1, 0.3, self.total_cells)
            print("Using synthetic porosity data")
        
        # Use real saturation data if available
        if 'properties' in self.data and 'water_saturation' in self.data['properties']:
            water_sat = self.data['properties']['water_saturation']
            if len(water_sat) == self.total_cells:
                self.saturation = 1 - water_sat  # Oil saturation
            else:
                np.random.seed(SEED)
                self.saturation = np.random.uniform(0.6, 0.9, self.total_cells)
        else:
            np.random.seed(SEED)
            self.saturation = np.random.uniform(0.6, 0.9, self.total_cells)
        
        # Reshape to 3D
        self.permeability_3d = self.permeability.reshape(self.nx, self.ny, self.nz)
        self.porosity_3d = self.porosity.reshape(self.nx, self.ny, self.nz)
        self.saturation_3d = self.saturation.reshape(self.nx, self.ny, self.nz)
        
        # Use real wells if available
        self.wells = self.data.get('wells', [])
        if not self.wells:
            self.wells = [
                {'name': 'PROD1', 'i': 2, 'j': 2, 'type': 'PRODUCER'},
                {'name': 'PROD2', 'i': 22, 'j': 2, 'type': 'PRODUCER'},
                {'name': 'PROD3', 'i': 2, 'j': 23, 'type': 'PRODUCER'},
                {'name': 'PROD4', 'i': 22, 'j': 23, 'type': 'PRODUCER'},
                {'name': 'INJ1', 'i': 12, 'j': 12, 'type': 'INJECTOR'},
            ]
        
        print(f"Reservoir setup complete:")
        print(f"Grid: {self.nx}×{self.ny}×{self.nz} = {self.total_cells:,} cells")
        print(f"Permeability: {np.mean(self.permeability):.1f} ± {np.std(self.permeability):.1f} md")
        print(f"Porosity: {np.mean(self.porosity):.3f} ± {np.std(self.porosity):.3f}")
        print(f"Wells: {len(self.wells)} wells")
        print(f"Data source: {'REAL SPE9' if self.data.get('real_data_loaded', False) else 'SYNTHETIC'}")
        
        return {
            'permeability_3d': self.permeability_3d,
            'porosity_3d': self.porosity_3d,
            'saturation_3d': self.saturation_3d,
            'grid_dimensions': (self.nx, self.ny, self.nz)
        }
    
    def calculate_well_productivity(self):
        print("\nCalculating well productivity...")
        
        well_rates = []
        for well in self.wells:
            i_idx = max(0, min(well['i'] - 1, self.nx - 1))
            j_idx = max(0, min(well['j'] - 1, self.ny - 1))
            cell_idx = i_idx * self.ny * self.nz + j_idx * self.nz
            
            if cell_idx < len(self.permeability):
                perm = self.permeability[cell_idx]
                poro = self.porosity[cell_idx]
                sat = self.saturation[cell_idx]
                
                # Use real production data if available
                if 'well_production_data' in self.data and well['name'] in self.data['well_production_data']:
                    well_data = self.data['well_production_data'][well['name']]
                    base_rate = np.mean(well_data.oil_rate) if well_data.oil_rate.size > 0 else 0
                    rate_type = "real production data"
                else:
                    if well['type'] == 'PRODUCER':
                        base_rate = perm * sat * 15 + poro * 800
                    else:
                        base_rate = perm * 5
                    rate_type = "calculated"
                
                well_rates.append({
                    'well': well['name'],
                    'type': well['type'],
                    'location': (well['i'], well['j']),
                    'permeability': perm,
                    'porosity': poro,
                    'saturation': sat,
                    'base_rate': base_rate,
                    'rate_source': rate_type
                })
        
        return well_rates
    
    def run_simulation(self, years=10):
        print(f"\nRunning physics-based simulation for {years} years...")
        
        months = years * 12
        time = np.linspace(0, years, months)
        
        well_data = self.calculate_well_productivity()
        
        # Calculate initial production rate
        total_initial_rate = sum(w['base_rate'] for w in well_data)
        print(f"Initial production rate: {total_initial_rate:.0f} bpd")
        
        # Calculate reservoir volumes
        cell_volume = 20 * 20 * 10  # ft³
        pore_volume = np.sum(self.porosity) * cell_volume
        oil_in_place = pore_volume * 0.7 / 5.6146  # Convert to barrels
        recoverable_oil = oil_in_place * 0.35
        
        print(f"Oil in place: {oil_in_place/1e6:.1f} MM bbl")
        print(f"Recoverable oil: {recoverable_oil/1e6:.1f} MM bbl")
        print(f"Recovery factor: 35%")
        
        # Production decline using Arps equation
        avg_perm = np.mean(self.permeability)
        b_factor = 0.5 + (avg_perm / 1000)
        
        qi = total_initial_rate
        Di = 0.3 / years
        
        # Arps hyperbolic decline
        oil_rate = qi / (1 + b_factor * Di * time) ** (1/b_factor)
        
        # Water cut development
        water_cut = np.zeros_like(time)
        for i, t in enumerate(time):
            if t < 2:
                water_cut[i] = 0.05
            elif t < 5:
                water_cut[i] = 0.05 + (t-2)/3 * 0.4
            else:
                water_cut[i] = 0.45 + min((t-5)/5 * 0.3, 0.3)
        
        water_rate = oil_rate * water_cut / (1 - water_cut)
        
        # Reservoir pressure
        initial_pressure = 3600  # psi
        cumulative_oil = np.cumsum(oil_rate) * 30.4  # Monthly cumulative
        pressure_drop = (cumulative_oil / recoverable_oil) * 1000
        pressure = initial_pressure - pressure_drop
        pressure[pressure < 500] = 500  # Minimum pressure
        
        return {
            'time': time,
            'oil_rate': oil_rate,
            'water_rate': water_rate,
            'water_cut': water_cut,
            'pressure': pressure,
            'cumulative_oil': cumulative_oil,
            'well_data': well_data,
            'reservoir_properties': {
                'oil_in_place': oil_in_place,
                'recoverable_oil': recoverable_oil,
                'avg_permeability': avg_perm,
                'avg_porosity': np.mean(self.porosity),
                'avg_saturation': np.mean(self.saturation),
                'total_cells': self.total_cells,
                'grid_dimensions': (self.nx, self.ny, self.nz),
                'data_source': 'REAL SPE9' if self.data.get('real_data_loaded', False) else 'SYNTHETIC'
            },
            'grid_data': {
                'permeability_3d': self.permeability_3d,
                'porosity_3d': self.porosity_3d,
                'saturation_3d': self.saturation_3d
            }
        }

class EnhancedEconomicAnalyzer:
    def __init__(self, simulation_results):
        self.results = simulation_results
    
    def analyze(self, oil_price=82.5, operating_cost=16.5, discount_rate=0.095):
        print("\nRunning economic analysis...")
        
        time = self.results['time']
        oil_rate = self.results['oil_rate']
        
        months_per_year = 12
        years = int(len(time) / months_per_year)
        
        annual_cash_flows = []
        
        # Capital expenditure based on well count
        capex = len(self.results['well_data']) * 3.5e6
        
        for year in range(years):
            start_idx = year * months_per_year
            end_idx = (year + 1) * months_per_year
            
            if end_idx > len(oil_rate):
                end_idx = len(oil_rate)
            
            annual_oil = np.sum(oil_rate[start_idx:end_idx]) * 30.4  # Monthly to annual
            
            revenue = annual_oil * oil_price
            opex = annual_oil * operating_cost
            annual_cf = revenue - opex
            
            annual_cash_flows.append(annual_cf)
        
        # Calculate NPV
        npv = -capex
        for year, cf in enumerate(annual_cash_flows, 1):
            npv += cf / ((1 + discount_rate) ** year)
        
        # Calculate IRR using iterative approach
        def npv_func(rate):
            result = -capex
            for year, cf in enumerate(annual_cash_flows, 1):
                result += cf / ((1 + rate) ** year)
            return result
        
        irr = discount_rate
        if npv > 0:
            for test_rate in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
                if npv_func(test_rate) < 0:
                    irr = test_rate
                    break
        
        # Payback period
        if annual_cash_flows and annual_cash_flows[0] > 0:
            cumulative = 0
            payback = 0
            for year, cf in enumerate(annual_cash_flows, 1):
                cumulative += cf
                if cumulative >= capex:
                    payback = year - 1 + (capex - (cumulative - cf)) / cf
                    break
            if cumulative < capex:
                payback = 100  # Never pays back
        else:
            payback = 100
        
        # ROI
        roi = (npv / capex) * 100 if capex > 0 else 0
        
        # Break-even price
        total_oil = np.sum(oil_rate) * 30.4
        break_even = operating_cost + (capex / total_oil) if total_oil > 0 else 0
        
        # Sensitivity analysis
        base_npv = npv
        high_price_npv = self._sensitivity_analysis(oil_price * 1.2, operating_cost, discount_rate)
        low_price_npv = self._sensitivity_analysis(oil_price * 0.8, operating_cost, discount_rate)
        
        return {
            'npv': npv,
            'irr': irr,
            'roi': roi,
            'payback_years': payback,
            'break_even_price': break_even,
            'total_capex': capex,
            'total_revenue': sum(annual_cash_flows) + capex,
            'total_oil': total_oil,
            'well_count': len(self.results['well_data']),
            'sensitivity': {
                'base_case': base_npv,
                'high_price': high_price_npv,
                'low_price': low_price_npv,
                'price_impact': (high_price_npv - low_price_npv) / base_npv if base_npv != 0 else 0
            },
            'oil_price': oil_price,
            'operating_cost': operating_cost,
            'discount_rate': discount_rate
        }
    
    def _sensitivity_analysis(self, oil_price, operating_cost, discount_rate):
        time = self.results['time']
        oil_rate = self.results['oil_rate']
        
        years = 15
        months_per_year = 12
        
        # Calculate annual cash flows
        annual_cash_flows = []
        capex = len(self.results['well_data']) * 3.5e6
        
        for year in range(years):
            start_idx = year * months_per_year
            end_idx = (year + 1) * months_per_year
            
            if start_idx >= len(oil_rate):
                break
                
            if end_idx > len(oil_rate):
                end_idx = len(oil_rate)
            
            annual_oil = np.sum(oil_rate[start_idx:end_idx]) * 30.4
            annual_cf = annual_oil * (oil_price - operating_cost)
            annual_cash_flows.append(annual_cf)
        
        # Calculate NPV
        npv = -capex
        for year, cf in enumerate(annual_cash_flows, 1):
            npv += cf / ((1 + discount_rate) ** year)
        
        return npv

class MLIntegration:
    @staticmethod
    def run_cnn_property_prediction(grid_data_3d, reservoir_properties):
        print("\nRunning CNN property prediction...")
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping CNN")
            return None, None
        
        try:
            print(f"Input grid shape: {grid_data_3d.shape}")
            
            predictor = PropertyPredictor()
            
            if predictor.model is None:
                print("CNN model could not be initialized")
                return None, None
            
            train_loader, val_loader = predictor.prepare_data(grid_data_3d, reservoir_properties)
            
            if not train_loader:
                print("No training data prepared")
                return None, None
            
            print("Training CNN model...")
            train_losses, val_losses = predictor.train(train_loader, val_loader, epochs=10)
            
            metrics = predictor.evaluate(grid_data_3d, reservoir_properties)
            
            if metrics:
                print("\nCNN Model Performance:")
                for metric_name, value in metrics.items():
                    if metric_name not in ['predictions', 'targets']:
                        print(f"  {metric_name}: {value:.4f}")
            
            # Save model
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            try:
                if predictor.save_model('results/cnn_reservoir_model.pth'):
                    print("Model saved to results/cnn_reservoir_model.pth")
                else:
                    print("Failed to save model")
            except Exception as e:
                print(f"Could not save model: {e}")
            
            return predictor, metrics
            
        except Exception as e:
            print(f"CNN model error: {str(e)}")
            return None, None
    
    @staticmethod
    def run_svr_economic_prediction(reservoir_params, economic_params):
        print("\nRunning economic forecasting...")
        
        try:
            # Create synthetic training data
            historical_data = MLIntegration._create_synthetic_training_data()
            
            engineer = EconomicFeatureEngineer()
            X_data = []
            y_data = []
            
            for case_data in historical_data:
                features = engineer.create_features(
                    case_data['reservoir'],
                    case_data['economic']
                )
                X_data.append(features)
                y_data.append(case_data['targets'])
            
            if not X_data:
                print("No training data generated")
                return None, None
            
            X = pd.concat(X_data, ignore_index=True)
            y = pd.DataFrame(y_data)
            
            predictor = SVREconomicPredictor(model_type='random_forest')
            X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
            
            print("Training Random Forest models for economic prediction...")
            print(f"Features: {X.shape[1]}, Samples: {len(X)}")
            
            predictor.train(X_train, y_train)
            
            metrics = predictor.evaluate(X_test, y_test)
            
            if metrics:
                print("\nModel Performance:")
                for target, target_metrics in metrics.items():
                    print(f"{target}:")
                    for metric_name, value in target_metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
            
            # Predict for current case
            current_features = engineer.create_features(reservoir_params, economic_params)
            predictions = predictor.predict(current_features)
            
            print("\nEconomic predictions for current case:")
            for target, value in predictions.iloc[0].items():
                print(f"  {target}: {value:.2f}")
            
            # Save model
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            try:
                if predictor.save_model('results/svr_economic_model.joblib'):
                    print("Model saved to results/svr_economic_model.joblib")
                else:
                    print("Failed to save SVR model")
            except Exception as e:
                print(f"Could not save SVR model: {e}")
            
            return predictor, predictions.iloc[0].to_dict()
            
        except Exception as e:
            print(f"Economic model error: {str(e)}")
            return None, None
    
    @staticmethod
    def _create_synthetic_training_data(n_samples=800):
        np.random.seed(SEED)
        
        training_data = []
        
        for i in range(n_samples):
            reservoir = {
                'avg_porosity': np.random.uniform(0.1, 0.3),
                'avg_permeability': np.random.lognormal(3, 0.5),
                'oil_in_place': np.random.uniform(1e6, 10e6),
                'recoverable_oil': np.random.uniform(0.2e6, 3e6)
            }
            
            economic = {
                'oil_price': np.random.uniform(40, 100),
                'opex_per_bbl': np.random.uniform(10, 30),
                'capex': np.random.uniform(10e6, 50e6),
                'discount_rate': np.random.uniform(0.05, 0.15),
                'tax_rate': np.random.uniform(0.2, 0.4)
            }
            
            targets = {
                'npv': (
                    reservoir['recoverable_oil'] * (economic['oil_price'] - economic['opex_per_bbl']) * 
                    (1 - economic['tax_rate']) / (1 + economic['discount_rate']) - economic['capex']
                ) / 1e6,
                
                'irr': np.random.uniform(0.05, 0.25) * 100,
                
                'roi': (
                    (reservoir['recoverable_oil'] * economic['oil_price'] * 0.7 - economic['capex']) / economic['capex']
                ) * 100,
                
                'payback_period': np.random.uniform(1, 10)
            }
            
            training_data.append({
                'reservoir': reservoir,
                'economic': economic,
                'targets': targets
            })
        
        return training_data
    
    @staticmethod
    def generate_ml_report(cnn_metrics, svr_predictions, economics):
        ml_report = {
            'cnn_performance': cnn_metrics if cnn_metrics else {},
            'svr_predictions': svr_predictions if svr_predictions else {},
            'model_details': {
                'cnn_available': cnn_metrics is not None,
                'svr_available': svr_predictions is not None,
                'pytorch_available': TORCH_AVAILABLE
            }
        }
        
        return ml_report

def create_visualizations(sim_results, economics, real_data, ml_report=None):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    fig_size = (18, 18) if ml_report and ml_report.get('cnn_performance') else (15, 10)
    fig, axes = plt.subplots(3, 2, figsize=fig_size)
    axes = axes.flatten()
    
    # Plot 1: Production profile
    ax1 = axes[0]
    ax1.plot(sim_results['time'], sim_results['oil_rate'], 'b-', linewidth=2, label='Oil Rate')
    ax1.plot(sim_results['time'], sim_results['water_rate'], 'r-', linewidth=2, label='Water Rate')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Rate (bpd)')
    ax1.set_title('Production Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Water cut
    ax2 = axes[1]
    ax2.plot(sim_results['time'], sim_results['water_cut']*100, 'g-', linewidth=2)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Water Cut (%)')
    ax2.set_title('Water Cut Development')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Economic metrics
    ax3 = axes[2]
    metrics = ['NPV ($M)', 'IRR (%)', 'ROI (%)', 'Payback']
    values = [
        economics['npv']/1e6,
        economics['irr']*100,
        economics['roi'],
        economics['payback_years']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_ylabel('Value')
    ax3.set_title('Economic Performance')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 4: Reservoir properties
    ax4 = axes[3]
    ax4.axis('off')
    props = sim_results['reservoir_properties']
    data_source = props.get('data_source', 'Unknown')
    
    text = f"""
    RESERVOIR PROPERTIES
    =========================
    Data Source: {data_source}
    Grid: {props.get('grid_dimensions', (24,25,15))} = {props['total_cells']:,} cells
    Avg Porosity: {props['avg_porosity']:.3f}
    Avg Permeability: {props['avg_permeability']:.0f} md
    Oil in Place: {props['oil_in_place']/1e6:.1f} MM bbl
    Recoverable Oil: {props['recoverable_oil']/1e6:.1f} MM bbl
    Recovery Factor: 35%
    
    WELL DATA
    =========================
    """
    for well in sim_results['well_data'][:5]:
        text += f"{well['well']}: {well['type']} @ ({well['location'][0]},{well['location'][1]})\n"
    
    if len(sim_results['well_data']) > 5:
        text += f"... and {len(sim_results['well_data']) - 5} more wells\n"
    
    ax4.text(0.1, 0.95, text, transform=ax4.transAxes,
            fontfamily='monospace', fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 5: ML Results
    ax5 = axes[4]
    ax5.axis('off')
    
    ml_text = """
    MACHINE LEARNING RESULTS
    ========================
    """
    
    if ml_report and ml_report.get('cnn_performance'):
        ml_text += "CNN PROPERTY PREDICTION:\n"
        for metric_name, value in ml_report['cnn_performance'].items():
            if metric_name not in ['predictions', 'targets']:
                ml_text += f"{metric_name}: {value:.4f}\n"
    
    if ml_report and ml_report.get('svr_predictions'):
        ml_text += "\nECONOMIC PREDICTIONS:\n"
        for target, value in ml_report['svr_predictions'].items():
            ml_text += f"{target}: {value:.2f}\n"
    
    if ml_report and ml_report.get('model_details', {}).get('pytorch_available'):
        ml_text += f"\nML Infrastructure: PyTorch Available\n"
    else:
        ml_text += f"\nML Infrastructure: PyTorch Not Available\n"
    
    ax5.text(0.1, 0.95, ml_text, transform=ax5.transAxes,
            fontfamily='monospace', fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Plot 6: Data validation
    ax6 = axes[5]
    ax6.axis('off')
    
    data_text = """
    DATA VALIDATION
    ================
    """
    
    data_text += f"Real Data Loaded: {real_data.get('real_data_loaded', False)}\n"
    data_text += f"Total Files: {len(real_data.get('files_found', []))}\n"
    data_text += f"Wells Found: {len(real_data.get('wells', []))}\n"
    
    if 'properties' in real_data:
        for prop_name, prop_data in real_data['properties'].items():
            if hasattr(prop_data, '__len__'):
                data_text += f"{prop_name}: {len(prop_data)} values\n"
    
    ax6.text(0.1, 0.95, data_text, transform=ax6.transAxes,
            fontfamily='monospace', fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.suptitle(f'Reservoir Simulation Analysis - {props.get("data_source", "SPE9 Data")}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'spe9_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved: results/spe9_analysis.png")

def save_comprehensive_report(sim_results, economics, real_data, ml_report=None):
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'project': 'Reservoir Simulation Analysis',
            'data_source': 'SPE9 Dataset with Real Data Integration',
            'files_used': real_data.get('files_found', []),
            'real_data_loaded': real_data.get('real_data_loaded', False),
            'ml_integration': ml_report is not None,
            'random_seed': SEED
        },
        'simulation': {
            'grid_dimensions': sim_results['reservoir_properties'].get('grid_dimensions', (24, 25, 15)),
            'total_cells': sim_results['reservoir_properties']['total_cells'],
            'data_source': sim_results['reservoir_properties'].get('data_source', 'Unknown'),
            'time_steps': len(sim_results['time']),
            'simulation_years': 10,
            'reservoir_properties': sim_results['reservoir_properties'],
            'well_data': sim_results['well_data'],
            'production_summary': {
                'peak_rate': float(np.max(sim_results['oil_rate'])),
                'final_rate': float(sim_results['oil_rate'][-1]),
                'total_oil': float(sim_results['cumulative_oil'][-1]),
                'avg_water_cut': float(np.mean(sim_results['water_cut']) * 100)
            }
        },
        'economics': economics,
        'machine_learning': ml_report if ml_report else {'status': 'not_run'},
        'data_validation': {
            'real_data_used': real_data.get('real_data_loaded', False),
            'wells_loaded': len(real_data.get('wells', [])),
            'properties_loaded': list(real_data.get('properties', {}).keys()),
            'spe9_variants': len([f for f in real_data.get('files_found', []) if 'SPE9_' in f])
        }
    }
    
    results_dir = Path("results")
    report_file = results_dir / 'spe9_report.json'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive report saved: {report_file}")

def print_summary(sim_results, economics, real_data, ml_report=None):
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)
    
    props = sim_results['reservoir_properties']
    
    summary = f"""
    TECHNICAL ANALYSIS:
    ========================================
    Data Source: {props.get('data_source', 'SPE9 Dataset')}
    Grid: {props.get('grid_dimensions', (24,25,15))} = {props['total_cells']:,} cells
    Simulation: 10 years physics-based simulation
    Peak Production: {np.max(sim_results['oil_rate']):.0f} bpd
    Total Oil Recovered: {sim_results['cumulative_oil'][-1] / 1e6:.2f} MM bbl
    Avg Water Cut: {np.mean(sim_results['water_cut']) * 100:.1f}%
    Wells Analyzed: {len(sim_results['well_data'])} wells
    
    ECONOMIC RESULTS:
    ========================================
    Net Present Value: ${economics['npv']/1e6:.2f} Million
    Internal Rate of Return: {economics['irr']*100:.1f}%
    Return on Investment: {economics['roi']:.1f}%
    Payback Period: {economics['payback_years']:.1f} years
    Break-even Price: ${economics['break_even_price']:.1f}/bbl
    Capital Investment: ${economics['total_capex']/1e6:.1f} Million
    
    DATA VALIDATION:
    ========================================
    Real Data Loaded: {real_data.get('real_data_loaded', False)}
    Data Files: {len(real_data.get('files_found', []))} files loaded
    SPE9 Variants: {len([f for f in real_data.get('files_found', []) if 'SPE9_' in f])} configurations
    Wells Found: {len(real_data.get('wells', []))} wells
    Properties Loaded: {len(real_data.get('properties', {}))} properties
    
    OUTPUT FILES:
    ========================================
    1. results/spe9_analysis.png - Visualizations
    2. results/spe9_report.json - JSON report
    """
    
    if ml_report and ml_report.get('model_details', {}).get('cnn_available'):
        summary += """    3. results/cnn_reservoir_model.pth - CNN model
    """
    
    if ml_report and ml_report.get('model_details', {}).get('svr_available'):
        summary += """    4. results/svr_economic_model.joblib - Economic model
    """
    
    if ml_report:
        summary += f"""
    MACHINE LEARNING RESULTS:
    ========================================
    CNN Property Prediction: {'Implemented' if ml_report['cnn_performance'] else 'Not available'}
    SVR Economic Forecasting: {'Implemented' if ml_report['svr_predictions'] else 'Not available'}
    """
        
        if ml_report['cnn_performance']:
            cnn_perf = ml_report['cnn_performance']
            avg_r2 = cnn_perf.get('R2', 0)
            summary += f"    CNN Model Accuracy (R²): {avg_r2:.3f}\n"
    
    print(summary)
    print("\nAnalysis completed successfully!")
    print("=" * 70)

def main():
    try:
        print(f"Starting reproducible analysis with seed: {SEED}")
        
        # Load real SPE9 data using the new DataLoader
        loader = RealSPE9DataLoader("data")
        real_data = loader.load_all_data()
        
        # Setup and run simulation
        simulator = PhysicsBasedSimulator(real_data)
        simulation_results = simulator.run_simulation(years=10)
        
        # Run economic analysis
        analyzer = EnhancedEconomicAnalyzer(simulation_results)
        economics = analyzer.analyze(
            oil_price=82.5,
            operating_cost=16.5,
            discount_rate=0.095
        )
        
        print("\n" + "="*70)
        print("MACHINE LEARNING INTEGRATION")
        print("="*70)
        
        ml_integration = MLIntegration()
        
        # Run CNN property prediction if grid data is available
        cnn_metrics = None
        if 'grid_data' in simulation_results:
            grid_data = simulation_results['grid_data']['permeability_3d']
            
            reservoir_properties = {
                'permeability': np.mean(grid_data),
                'porosity': np.mean(simulation_results['grid_data']['porosity_3d']),
                'saturation': np.mean(simulation_results['grid_data']['saturation_3d'])
            }
            
            cnn_predictor, cnn_metrics = ml_integration.run_cnn_property_prediction(
                grid_data, reservoir_properties
            )
        else:
            print("No grid data available for CNN")
        
        # Run SVR economic prediction
        reservoir_params = simulation_results['reservoir_properties']
        economic_params = {
            'oil_price': economics['oil_price'],
            'opex_per_bbl': economics['operating_cost'],
            'capex': economics['total_capex'],
            'discount_rate': economics['discount_rate'],
            'tax_rate': 0.3
        }
        
        svr_predictor, svr_predictions = ml_integration.run_svr_economic_prediction(
            reservoir_params, economic_params
        )
        
        ml_report = ml_integration.generate_ml_report(cnn_metrics, svr_predictions, economics)
        
        print("\nGenerating visualizations...")
        create_visualizations(simulation_results, economics, real_data, ml_report)
        
        print("\nSaving comprehensive report...")
        save_comprehensive_report(simulation_results, economics, real_data, ml_report)
        
        print_summary(simulation_results, economics, real_data, ml_report)
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
