"""
Main simulation runner for SPE9 reservoir simulation
"""

import os
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Controls and executes reservoir simulation"""
    
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """Initialize simulation runner with configuration"""
        self.config = self.load_config(config_path)
        self.simulator = self.detect_simulator()
        self.results_dir = "results/simulation_output"
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load simulation configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
    def detect_simulator(self) -> str:
        """Detect available reservoir simulator"""
        simulators = ['flow', 'eclipse', 'intersect']
        
        for simulator in simulators:
            try:
                subprocess.run([simulator, '--version'], 
                             capture_output=True, check=False)
                logger.info(f"Detected simulator: {simulator}")
                return simulator
            except FileNotFoundError:
                continue
                
        logger.warning("No simulator found. Using 'flow' as default")
        return 'flow'
        
    def validate_inputs(self) -> bool:
        """Validate all input files before simulation"""
        required_files = [
            "data/SPE9.DATA",
            "data/SPE9_GRID.INC",
            "data/SPE9_PORO.INC",
            "data/SPE9_PVT.INC",
            "data/SPE9_SATURATION_TABLES.INC"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"Missing required file: {file_path}")
                return False
                
        logger.info("All input files validated successfully")
        return True
        
    def prepare_environment(self) -> None:
        """Prepare simulation environment"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create symbolic links to data files if needed
        if not os.path.exists("SPE9.DATA"):
            os.symlink("data/SPE9.DATA", "SPE9.DATA")
            
    def run_simulation(self) -> bool:
        """Execute the reservoir simulation"""
        if not self.validate_inputs():
            return False
            
        self.prepare_environment()
        
        logger.info(f"Starting SPE9 simulation at {datetime.now()}")
        logger.info(f"Simulator: {self.simulator}")
        logger.info(f"Grid: {self.config['grid']['dimensions']}")
        logger.info(f"Wells: {self.config['wells']['total']}")
        
        try:
            cmd = [self.simulator, "SPE9.DATA"]
            
            with open(f"{self.results_dir}/simulation.log", "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Stream output to log file and console
                for line in process.stdout:
                    log_file.write(line)
                    if "TIME" in line or "SUMMARY" in line:
                        logger.info(line.strip())
                        
                process.wait()
                
                if process.returncode == 0:
                    logger.info("Simulation completed successfully")
                    self._copy_results()
                    return True
                else:
                    error_output = process.stderr.read()
                    logger.error(f"Simulation failed: {error_output}")
                    return False
                    
        except Exception as e:
            logger.error(f"Simulation execution error: {e}")
            return False
            
    def _copy_results(self) -> None:
        """Copy simulation results to results directory"""
        result_files = [
            "SPE9.UNRST",
            "SPE9.SMSPEC",
            "SPE9.INIT",
            "SPE9.EGRID",
            "SPE9.PRT"
        ]
        
        for file in result_files:
            if os.path.exists(file):
                os.rename(file, f"{self.results_dir}/{file}")
                logger.info(f"Saved: {file}")
                
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get simulation information and statistics"""
        return {
            "simulator": self.simulator,
            "config": self.config,
            "results_directory": self.results_dir,
            "timestamp": datetime.now().isoformat()
        }
