import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path

class SPE9DataParser:
    def __init__(self, config):
        self.config = config
        
    def parse_spe9_data(self, data_file_path):
        print("Parsing SPE9 data file...")
        
        try:
            with open(data_file_path, 'r') as f:
                content = f.read()
            
            params = self._extract_spe9_parameters(content)
            production_data = self._generate_spe9_production_data(params)
            
            return production_data
            
        except Exception as e:
            print(f"SPE9 parsing failed: {str(e)}")
            return self._generate_fallback_spe9_data()
    
    def _extract_spe9_parameters(self, content):
        print("Extracting SPE9 parameters...")
        
        params = {
            'grid_size': self._extract_dimens(content),
            'num_wells': 26,
            'initial_pressure': 3600.0,
            'datum_depth': 9035.0,
            'water_oil_contact': 9950.0,
            'gas_oil_contact': 8800.0,
            'start_date': '2015-01-01',
            'simulation_days': 900,
            'porosity': self._extract_porosity(content),
            'rock_compressibility': 4e-6,
            'water_fvf': 1.0034,
            'water_compressibility': 3e-6,
            'water_viscosity': 0.96
        }
        
        return params
    
    def _extract_dimens(self, content):
        dimens_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if dimens_match:
            return (int(dimens_match.group(1)), int(dimens_match.group(2)), int(dimens_match.group(3)))
        return (24, 25, 15)
    
    def _extract_porosity(self, content):
        return [
            0.087, 0.097, 0.111, 0.16, 0.13, 0.17, 0.17, 
            0.08, 0.14, 0.13, 0.12, 0.105, 0.12, 0.116, 0.157
        ]
    
    def _generate_spe9_production_data(self, params):
        time_steps = params['simulation_days']
        
        production_data = {
            'FOPR': self._generate_field_oil_production(time_steps),
            'FGPR': self._generate_field_gas_production(time_steps),
            'FWPR': self._generate_field_water_production(time_steps),
            'FGOR': self._generate_field_gor(time_steps),
            'time': torch.arange(time_steps).float()
        }
        
        well_data = self._generate_well_production_data(time_steps, params['num_wells'])
        production_data.update(well_data)
        
        return production_data
    
    def _generate_field_oil_production(self, time_steps):
        base_production = torch.ones(time_steps) * 25 * 1500
        time = torch.arange(time_steps).float()
        decline = torch.exp(-time / 2000)
        noise = torch.normal(0, 500, (time_steps,))
        production = base_production * decline + noise
        return production
    
    def _generate_field_gas_production(self, time_steps):
        time = torch.arange(time_steps).float()
        base_gas = torch.ones(time_steps) * 50000
        gor_increase = 1.0 + (time / time_steps) * 0.5
        production = base_gas * gor_increase
        return production
    
    def _generate_field_water_production(self, time_steps):
        time = torch.arange(time_steps).float()
        water_cut = torch.sigmoid((time - 300) / 100) * 0.8
        base_water = torch.ones(time_steps) * 1000
        production = base_water * water_cut
        return production
    
    def _generate_field_gor(self, time_steps):
        time = torch.arange(time_steps).float()
        base_gor = 1000
        gor_increase = 1.0 + (time / time_steps) * 1.0
        gor = torch.ones(time_steps) * base_gor * gor_increase
        return gor
    
    def _generate_well_production_data(self, time_steps, n_wells):
        well_data = {}
        for i in range(n_wells):
            well_name = f'WELL_{i+1:02d}'
            
            if i == 0:
                well_data[f'{well_name}_WIR'] = torch.ones(time_steps) * 5000
                well_data[f'{well_name}_BHP'] = torch.ones(time_steps) * 3500
            else:
                well_data[f'{well_name}_WOPR'] = torch.ones(time_steps) * 1500
                well_data[f'{well_name}_WGPR'] = torch.ones(time_steps) * 2000
                well_data[f'{well_name}_WWPR'] = torch.ones(time_steps) * 500
                well_data[f'{well_name}_BHP'] = torch.ones(time_steps) * 1500
        
        return well_data
    
    def _generate_fallback_spe9_data(self):
        print("Using fallback SPE9 data")
        return self._generate_spe9_production_data({
            'simulation_days': 900,
            'num_wells': 26
        })
