
class SPE9Preprocessor:
    """Preprocessor for real SPE9 dataset."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.config.validate()
        
        # Use real SPE9 parser
        from .spe9_parser import SPE9Parser
        self.parser = SPE9Parser()
        
        # Scalers
        self.input_scaler = None
        self.target_scaler = None
        self.statistics = {}
        
        logger.info(f"Preprocessor initialized for real SPE9 data")
    
    def load_real_spe9_data(self, data_dir: Path) -> Dict[str, Any]:
        """Load real SPE9 dataset."""
        logger.info(f"Loading real SPE9 data from {data_dir}")
        
        # Parse real SPE9 data
        dataset = self.parser.load_spe9_dataset(data_dir)
        
        # Extract time series data
        summary_df = dataset.get('summary', pd.DataFrame())
        
        if summary_df.empty:
            raise ValueError("No summary data found in SPE9 dataset")
        
        logger.info(f"Loaded SPE9 data: {summary_df.shape}")
        logger.info(f"Columns: {list(summary_df.columns)}")
        
        return {
            'summary': summary_df,
            'grid': dataset.get('grid', pd.DataFrame()),
            'properties': dataset.get('properties', {}),
            'metadata': {
                'source': 'SPE9 Benchmark',
                'grid_size': '24×25×15',
                'simulation_period': '10 years',
                'wells': '5 producers, 4 injectors',
            }
        }
    
    def preprocess(self, data_dir: Path) -> Dict[str, np.ndarray]:
        """
        Preprocess real SPE9 data.
        
        Args:
            data_dir: Directory containing SPE9 DATA files
            
        Returns:
            Dictionary with processed data splits
        """
        logger.info("Preprocessing real SPE9 data")
        
        # Load real data
        dataset = self.load_real_spe9_data(data_dir)
        summary_df = dataset['summary']
        
        # Store dataset for reference
        self.dataset = dataset
        
        # Prepare features and targets based on SPE9 variables
        X_columns = self._select_input_features(summary_df)
        y_columns = self._select_target_features(summary_df)
        
        X = summary_df[X_columns].values
        y = summary_df[y_columns].values
        
        # Handle missing values
        X = self._handle_missing_values(X)
        y = self._handle_missing_values(y)
        
        # Scale data
        X_scaled, y_scaled = self._scale_data(X, y)
        
        # Create sequences
        sequences = self._create_sequences(X_scaled, y_scaled)
        
        # Split data
        splits = self._split_data(sequences)
        
        # Store statistics
        self._compute_statistics(summary_df, X_scaled, y_scaled, splits)
        
        logger.info("SPE9 data preprocessing completed")
        return splits
    
    def _select_input_features(self, df: pd.DataFrame) -> List[str]:
        """Select input features from SPE9 data."""
        available_columns = set(df.columns)
        
        # Prioritize real SPE9 variables
        spe9_variables = [
            'FOPR', 'FWPR', 'FGPR',      # Production rates
            'FWIR', 'FGIR',              # Injection rates
            'WBHP:PROD1', 'WBHP:PROD2',  # Producer BHPs
            'WBHP:INJ1', 'WBHP:INJ2',    # Injector BHPs
            'TIME',                      # Time
        ]
        
        # Use available variables
        selected = [col for col in spe9_variables if col in available_columns]
        
        # If not enough variables, use all available
        if len(selected) < 3:
            selected = list(df.columns)[:min(10, len(df.columns))]
        
        logger.info(f"Selected input features: {selected}")
        return selected
    
    def _select_target_features(self, df: pd.DataFrame) -> List[str]:
        """Select target features from SPE9 data."""
        available_columns = set(df.columns)
        
        # Common prediction targets in reservoir engineering
        target_candidates = [
            'FOPR',    # Oil production rate (primary target)
            'FOPT',    # Cumulative oil production
            'WBHP:PROD1',  # Key well pressure
            'FWPR',    # Water production rate
        ]
        
        # Use available targets
        selected = [col for col in target_candidates if col in available_columns]
        
        # If no specific targets, use first column
        if not selected:
            selected = [df.columns[0]]
        
        logger.info(f"Selected target features: {selected}")
        return selected
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded SPE9 dataset."""
        if not hasattr(self, 'dataset'):
            return {}
        
        info = {
            'data_source': 'SPE9 Benchmark Dataset',
            'summary_shape': self.dataset['summary'].shape if 'summary' in self.dataset else None,
            'grid_size': self.dataset.get('metadata', {}).get('grid_size'),
            'simulation_period': self.dataset.get('metadata', {}).get('simulation_period'),
            'available_variables': list(self.dataset['summary'].columns) if 'summary' in self.dataset else [],
            'reservoir_properties': self.dataset.get('properties', {}),
        }
        
        return info
