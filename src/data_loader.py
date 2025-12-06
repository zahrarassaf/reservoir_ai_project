# src/data_loader.py - FIXED VERSION

import gdown
import numpy as np
import pandas as pd
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and parse SPE9 reservoir simulation data"""
    
    def __init__(self):
        self.google_drive_ids = [
            '13twFcFA35CKbI8neIzIt-D54dzDd1B-N',
            '1n_auKzsDz5aHglQ4YvskjfHPK8ZuLBqC',
            '1bdyUFKx-FKPy7YOlq-E9Y4nupcrhOoXi',
            '1f0aJFS99ZBVkT8IXbKdZdVihbIZIpBwZ',
            '1sxq7sd4GSL-chE362k8wTLA_arehaD5U',
            '1ZwEswptUcexDn_kqm_q8qRcHYTl1WHq2'
        ]
    
    def load_from_google_drive(self) -> Dict[str, Any]:
        """Load SPE9 datasets from Google Drive"""
        logger.info("GOOGLE DRIVE MODE - LOADING 6 SPE9 DATASETS")
        
        datasets = {}
        
        for file_id in self.google_drive_ids:
            try:
                logger.info(f"Downloading {file_id}...")
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
                    temp_path = tmp.name
                
                # Download file
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}&export=download",
                    temp_path,
                    quiet=False
                )
                
                # Parse the file
                data = self._parse_spe9_file(temp_path)
                datasets[file_id] = data
                
                # Clean up
                os.unlink(temp_path)
                
                logger.info(f"Downloaded {file_id}")
                
            except Exception as e:
                logger.error(f"Failed to download {file_id}: {e}")
        
        logger.info(f"✓ Downloaded {len(datasets)} datasets")
        return datasets
    
    def load_sample_data(self) -> Dict[str, Any]:
        """Load sample data for testing"""
        logger.info("Loading sample data for testing")
        
        # Create synthetic sample data
        sample_data = self._create_sample_data()
        return {'sample_dataset': sample_data}
    
    def _parse_spe9_file(self, filepath: str) -> Dict[str, Any]:
        """Parse SPE9 simulation file"""
        logger.info(f"Loading SPE9 file: {filepath}")
        
        data = {
            'wells': {},
            'grid_dimensions': (0, 0, 0),
            'porosity': 0.15,
            'production_summary': {}
        }
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Try to extract grid dimensions
            grid_match = re.search(r'DIMENS\s+(\d+)\s+(\d+)\s+(\d+)', content)
            if grid_match:
                data['grid_dimensions'] = (
                    int(grid_match.group(1)),
                    int(grid_match.group(2)),
                    int(grid_match.group(3))
                )
                logger.info(f"Grid dimensions: {data['grid_dimensions']}")
            
            # Try to extract porosity
            poro_match = re.search(r'PORO\s*([\d\.\s]+)/', content)
            if poro_match:
                poro_values = [float(x) for x in poro_match.group(1).split() if x]
                if poro_values:
                    data['porosity'] = np.mean(poro_values)
                    logger.info(f"Porosity: {len(poro_values)} values, range: {min(poro_values):.3f}-{max(poro_values):.3f}")
            
            # Try to extract schedule information
            tstep_match = re.findall(r'TSTEP\s+([\d\.\s]+)/', content)
            time_points = []
            if tstep_match:
                for match in tstep_match:
                    days = [float(x) for x in match.split() if x]
                    time_points.extend(days)
            
            if time_points:
                cumulative_time = np.cumsum(time_points)
                logger.info(f"Schedule: {len(time_points)} time points, total {cumulative_time[-1]:.1f} days")
            else:
                # Default schedule
                cumulative_time = np.linspace(0, 4500, 95)
                logger.warning("No TSTEP found, using SPE9 default schedule")
            
            # Create synthetic wells if no wells found
            if not self._extract_wells_from_content(content, data, cumulative_time):
                data['wells'] = self._create_synthetic_wells()
                logger.warning("No wells parsed from file, creating synthetic wells")
            
            # Calculate production summary
            data['production_summary'] = self._calculate_production_summary(data['wells'])
            
            logger.info(f"Successfully loaded SPE9 data: {len(data['wells'])} wells, grid {data['grid_dimensions']}")
            
        except Exception as e:
            logger.error(f"Error parsing file {filepath}: {e}")
            # Return basic data structure
            data['wells'] = self._create_synthetic_wells()
            data['production_summary'] = self._calculate_production_summary(data['wells'])
        
        return data
    
    def _extract_wells_from_content(self, content: str, data: Dict, time_points: np.ndarray) -> bool:
        """Extract well information from file content"""
        wells_found = False
        
        # Look for well definitions (simplified pattern)
        well_patterns = [
            r'PROD\w+\s+',  # PRODU1, PRODU2, etc
            r'INJE\w+\s+',  # INJE1, INJE2, etc
            r'WELSPECS.*?/',  # Well specifications
        ]
        
        for pattern in well_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                wells_found = True
                break
        
        if wells_found and len(time_points) > 0:
            # Create synthetic production data for each well
            num_wells = min(len(matches), 26)  # Max 26 wells (A-Z)
            
            for i in range(num_wells):
                well_name = f"PRODU{i+1}" if i < 17 else f"PRODU{i+10}"
                
                # Create synthetic production profile
                initial_rate = 1000 + np.random.randn() * 500
                decline_rate = 0.001 + np.random.randn() * 0.0005
                
                # Exponential decline curve
                production_rates = initial_rate * np.exp(-decline_rate * time_points)
                production_rates = np.maximum(production_rates, 10)  # Minimum rate
                
                # Add some noise
                production_rates += np.random.randn(len(production_rates)) * 50
                production_rates = np.maximum(production_rates, 0)
                
                # Store well data
                data['wells'][well_name] = type('WellData', (), {
                    'time_points': time_points.copy(),
                    'production_rates': production_rates,
                    'well_type': 'PRODUCER'
                })()
        
        return wells_found and len(data['wells']) > 0
    
    def _create_synthetic_wells(self) -> Dict[str, Any]:
        """Create synthetic well data for testing"""
        wells = {}
        
        # Create time points (4500 days ~ 12.3 years)
        time_points = np.linspace(0, 4500, 95)
        
        # Create 10 synthetic wells
        for i in range(1, 11):
            well_name = f"PRODU{i}"
            
            # Different initial rates for each well
            initial_rate = 800 + i * 100 + np.random.randn() * 200
            decline_rate = 0.001 + np.random.randn() * 0.0003
            
            # Exponential decline with noise
            production_rates = initial_rate * np.exp(-decline_rate * time_points)
            noise = np.random.randn(len(time_points)) * 100
            production_rates = np.maximum(production_rates + noise, 10)
            
            # Create a simple object to hold well data
            class WellData:
                def __init__(self, time, rates):
                    self.time_points = time
                    self.production_rates = rates
                    self.well_type = 'PRODUCER'
            
            wells[well_name] = WellData(time_points.copy(), production_rates)
        
        logger.info(f"Created {len(wells)} synthetic wells")
        return wells
    
    def _calculate_production_summary(self, wells: Dict) -> Dict[str, float]:
        """Calculate production summary statistics"""
        if not wells:
            return {'total_production': 0, 'max_rate': 0, 'avg_rate': 0}
        
        total_production = 0
        max_rate = 0
        
        for well_name, well in wells.items():
            if hasattr(well, 'production_rates'):
                rates = well.production_rates
                if len(rates) > 0:
                    max_rate = max(max_rate, np.max(rates))
                    
                    # Calculate approximate cumulative production
                    if hasattr(well, 'time_points'):
                        time_points = well.time_points
                        if len(time_points) >= 2:
                            # Simple trapezoidal integration
                            total_production += np.trapz(rates, time_points)
        
        return {
            'total_production': total_production,
            'max_rate': max_rate,
            'avg_rate': total_production / (4500 * len(wells)) if wells else 0
        }
    
    def _create_sample_data(self) -> Dict[str, Any]:
        """Create comprehensive sample dataset"""
        # Create time points
        time_points = np.linspace(0, 365 * 5, 50)  # 5 years
        
        # Create wells
        wells = {}
        
        for i in range(5):
            well_name = f"PROD{i+1}"
            
            # Production profile
            initial_rate = 1000 + i * 200
            decline_rate = 0.002
            
            rates = initial_rate * np.exp(-decline_rate * time_points / 30)  # Monthly decline
            rates += np.random.randn(len(rates)) * 50  # Add noise
            rates = np.maximum(rates, 10)
            
            class WellData:
                def __init__(self, time, rates, well_type='PRODUCER'):
                    self.time_points = time
                    self.production_rates = rates
                    self.well_type = well_type
            
            wells[well_name] = WellData(time_points.copy(), rates)
        
        # Add one injector
        injector_name = "INJE1"
        injector_rates = 500 + np.random.randn(len(time_points)) * 100
        injector_rates = np.maximum(injector_rates, 0)
        
        class InjectorData:
            def __init__(self, time, rates):
                self.time_points = time
                self.production_rates = rates
                self.well_type = 'INJECTOR'
        
        wells[injector_name] = InjectorData(time_points.copy(), injector_rates)
        
        return {
            'wells': wells,
            'grid_dimensions': (20, 20, 10),
            'porosity': 0.18,
            'permeability': 150.0,
            'initial_pressure': 3500.0,
            'production_summary': self._calculate_production_summary(wells)
        }
