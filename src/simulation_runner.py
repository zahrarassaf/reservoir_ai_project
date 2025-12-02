"""
Main simulation runner for SPE9 reservoir simulation
"""

import os
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Controls and executes reservoir simulation"""
    
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """Initialize simulation runner with configuration"""
        self.config = self.load_config(config_path)
        self.simulator = self.detect_simulator()
        self.results_dir = Path("results/simulation_output")
        
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
                result = subprocess.run([simulator, '--version'], 
                                      capture_output=True, text=True, check=False)
                if result.returncode == 0 or 'version' in result.stdout.lower():
                    logger.info(f"Detected simulator: {simulator}")
                    return simulator
            except FileNotFoundError:
                continue
                
        logger.warning("No simulator found in PATH. Using 'flow' as default")
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
                
        logger.info("✅ All input files validated successfully")
        return True
        
    def prepare_environment(self) -> None:
        """Prepare simulation environment"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Prepared environment: results in {self.results_dir}")
        
    def run_simulation(self) -> bool:
        """Execute the reservoir simulation"""
        if not self.validate_inputs():
            logger.error("Input validation failed. Cannot run simulation.")
            return False
            
        self.prepare_environment()
        
        logger.info("=" * 60)
        logger.info("🚀 Starting SPE9 Reservoir Simulation")
        logger.info("=" * 60)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Simulator: {self.simulator}")
        logger.info(f"Grid: {self.config['grid']['dimensions']}")
        logger.info(f"Total Cells: {self.config['grid']['total_cells']}")
        logger.info(f"Wells: {self.config['wells']['total']}")
        logger.info(f"Simulation Period: {self.config['simulation_period']['total_days']} days")
        logger.info("=" * 60)
        
        try:
            # Change to data directory for simulation
            original_dir = os.getcwd()
            data_dir = Path("data")
            
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return False
                
            os.chdir(data_dir)
            
            # Prepare command
            cmd = [self.simulator, "SPE9.DATA"]
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run simulation
            log_file_path = self.results_dir.parent / "simulation.log"
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"SPE9 Simulation Log - {datetime.now()}\n")
                log_file.write("=" * 60 + "\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write("=" * 60 + "\n\n")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to log file and console
                logger.info("Simulation running...")
                for line in process.stdout:
                    log_file.write(line)
                    # Log important events
                    if "TIME" in line and "REPORT" in line:
                        logger.info(f"Simulation progress: {line.strip()}")
                    elif "ERROR" in line or "WARNING" in line:
                        logger.warning(line.strip())
                        
                # Wait for process to complete
                process.wait()
                
                # Capture stderr
                stderr_output = process.stderr.read()
                if stderr_output:
                    log_file.write("\n=== STDERR ===\n")
                    log_file.write(stderr_output)
                    logger.warning(f"Simulator stderr: {stderr_output[:200]}...")
                    
            # Return to original directory
            os.chdir(original_dir)
            
            if process.returncode == 0:
                logger.info("✅ Simulation completed successfully")
                self._copy_results()
                return True
            else:
                logger.error(f"❌ Simulation failed with return code: {process.returncode}")
                if stderr_output:
                    logger.error(f"Error output: {stderr_output[:500]}")
                return False
                
        except FileNotFoundError as e:
            logger.error(f"❌ Simulator not found: {self.simulator}")
            logger.error(f"Please install {self.simulator} or add it to PATH")
            return False
        except Exception as e:
            logger.error(f"❌ Simulation execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _copy_results(self) -> None:
        """Copy simulation results to results directory"""
        data_dir = Path("data")
        result_files = [
            "SPE9.UNRST",
            "SPE9.SMSPEC", 
            "SPE9.INIT",
            "SPE9.EGRID",
            "SPE9.PRT",
            "SPE9.LOG"
        ]
        
        copied = 0
        for file_name in result_files:
            src = data_dir / file_name
            if src.exists():
                dst = self.results_dir / file_name
                import shutil
                shutil.move(str(src), str(dst))
                copied += 1
                logger.info(f"📄 Saved: {file_name}")
                
        logger.info(f"📁 Total files saved: {copied}")
        
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get simulation information and statistics"""
        return {
            "simulator": self.simulator,
            "config": self.config,
            "results_directory": str(self.results_dir),
            "timestamp": datetime.now().isoformat(),
            "author": self.config.get('simulation', {}).get('author', 'Unknown')
        }
