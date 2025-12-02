"""
Simulation Runner Module
"""

import numpy as np
from datetime import datetime

class SimulationRunner:
    """Run reservoir simulations."""
    
    def __init__(self, reservoir_data, simulation_config=None, grid_config=None):
        self.data = reservoir_data
        self.config = simulation_config or {}
        self.grid = grid_config or {}
    
    def run(self):
        """Run simulation."""
        # This should be implemented based on your physics model
        # For now, return a basic structure
        
        time_steps = self.config.get('time_steps', 365)
        
        return {
            'metadata': {
                'simulation_date': datetime.now().isoformat(),
                'grid_dimensions': self.data.get('grid_dimensions', (24, 25, 15)),
                'time_steps': time_steps,
                'simulation_type': 'physics_based'
            },
            'time_series': {
                'time_steps': list(range(time_steps))
            },
            'production': self._generate_production_data(time_steps),
            'wells': self.data.get('wells', [])
        }
    
    def _generate_production_data(self, n_steps):
        """Generate production data."""
        time = np.arange(n_steps)
        
        oil = 1000 * np.exp(-0.0015 * time)
        water = 200 * (1 + 0.002 * time / n_steps)
        gas = oil * 500 / 1000
        
        return {
            'oil': oil.tolist(),
            'water': water.tolist(),
            'gas': gas.tolist()
        }
