# src/spe9_processor.py
import numpy as np
import pandas as pd
import os
from typing import Dict, List
from .economics import WellProductionData

class SPE9Processor:
    def create_spe9_benchmark_data(self) -> Dict:
        """
        Create synthetic data that mimics SPE9 benchmark reservoir
        SPE9 is a 3-phase, 3D black oil simulation with:
        - Grid: 24x25x15 (9000 cells)
        - Wells: 6 producers, 4 injectors
        - Time: 900 days
        - Heterogeneous properties
        """
        
        # Create time points (30 reporting periods over 900 days)
        time_points = np.linspace(0, 900, 30)
        
        wells = {}
        
        # Producer wells with realistic decline
        producer_locations = [
            {'name': 'PROD_01', 'i': 4, 'j': 4, 'k_range': (1, 8)},
            {'name': 'PROD_02', 'i': 9, 'j': 4, 'k_range': (1, 8)},
            {'name': 'PROD_03', 'i': 14, 'j': 4, 'k_range': (1, 8)},
            {'name': 'PROD_04', 'i': 19, 'j': 4, 'k_range': (1, 8)},
            {'name': 'PROD_05', 'i': 4, 'j': 21, 'k_range': (1, 8)},
            {'name': 'PROD_06', 'i': 19, 'j': 21, 'k_range': (1, 8)}
        ]
        
        for prod in producer_locations:
            # Base production profile with hyperbolic decline
            qi = np.random.uniform(600, 1200)  # Initial rate
            Di = np.random.uniform(0.001, 0.003)  # Initial decline
            b = np.random.uniform(0.8, 1.2)  # b-factor
            
            time_normalized = time_points / 365  # Convert to years
            oil_rate = qi / (1 + b * Di * time_normalized) ** (1/b)
            
            # Add noise and water breakthrough
            noise = 0.1 * np.random.randn(len(time_points))
            oil_rate = oil_rate * (1 + noise)
            oil_rate = np.maximum(oil_rate, 50)  # Economic limit
            
            # Gas and water production
            gor = np.random.uniform(400, 600)  # Gas-oil ratio (scf/stb)
            wor_start = np.random.uniform(0.05, 0.15)
            wor_end = np.random.uniform(0.3, 0.6)
            wor = np.linspace(wor_start, wor_end, len(time_points))
            
            gas_rate = oil_rate * gor
            water_rate = oil_rate * wor
            
            wells[prod['name']] = WellProductionData(
                time_points=time_points,
                oil_rate=oil_rate,
                gas_rate=gas_rate,
                water_rate=water_rate,
                bottomhole_pressure=3000 - 0.5 * time_points + 50 * np.random.randn(len(time_points)),
                well_type='PRODUCER'
            )
        
        # Injector wells
        injector_locations = [
            {'name': 'INJ_01', 'i': 4, 'j': 12, 'k_range': (1, 15)},
            {'name': 'INJ_02', 'i': 9, 'j': 12, 'k_range': (1, 15)},
            {'name': 'INJ_03', 'i': 14, 'j': 12, 'k_range': (1, 15)},
            {'name': 'INJ_04', 'i': 19, 'j': 12, 'k_range': (1, 15)}
        ]
        
        for inj in injector_locations:
            injection_rate = np.random.uniform(1500, 2500)
            water_rate = injection_rate * np.ones(len(time_points)) * (1 + 0.05 * np.random.randn(len(time_points)))
            
            wells[inj['name']] = WellProductionData(
                time_points=time_points,
                oil_rate=np.zeros(len(time_points)),
                water_rate=water_rate,
                bottomhole_pressure=4000 + 20 * np.random.randn(len(time_points)),
                well_type='INJECTOR'
            )
        
        # Grid properties
        nx, ny, nz = 24, 25, 15
        n_cells = nx * ny * nz
        
        # Create heterogeneous porosity field
        porosity = np.random.uniform(0.15, 0.25, n_cells)
        
        # Create log-normal permeability field
        permeability_x = np.random.lognormal(mean=2.5, sigma=0.8, size=n_cells)
        permeability_y = permeability_x * np.random.uniform(0.8, 1.2, n_cells)
        permeability_z = permeability_x * np.random.uniform(0.1, 0.3, n_cells)
        
        # Reservoir properties
        reservoir_props = {
            'initial_pressure': 3600,  # psi at datum
            'datum_depth': 8100,  # ft
            'temperature': 180,  # °F
            'initial_water_saturation': 0.25,
            'residual_oil_saturation': 0.25,
            'residual_gas_saturation': 0.05,
            'rock_compressibility': 3e-6,  # 1/psi
            'reference_pressure': 3600  # psi
        }
        
        # Fluid properties
        fluid_props = {
            'oil_fvf': 1.2,  # rb/stb
            'gas_fvf': 0.005,  # rb/scf
            'water_fvf': 1.0,  # rb/stb
            'oil_viscosity': 0.8,  # cp
            'gas_viscosity': 0.02,  # cp
            'water_viscosity': 0.5  # cp
        }
        
        return {
            'wells': wells,
            'grid': {
                'dimensions': (nx, ny, nz),
                'porosity': porosity,
                'permeability_x': permeability_x,
                'permeability_y': permeability_y,
                'permeability_z': permeability_z,
                'net_to_gross': np.random.uniform(0.7, 0.9, n_cells)
            },
            'reservoir': reservoir_props,
            'fluids': fluid_props,
            'metadata': {
                'name': 'SPE9 Benchmark Reservoir',
                'description': 'Synthetic data mimicking SPE9 comparative solution project',
                'grid_cells': n_cells,
                'producers': 6,
                'injectors': 4,
                'simulation_period': 900,
                'time_steps': 30
            }
        }
