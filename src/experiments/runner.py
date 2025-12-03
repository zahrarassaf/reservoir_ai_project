# src/experiments/runner.py (آپدیت شده)
class SPE9ExperimentRunner:
    """Experiment runner specifically for SPE9 dataset."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # SPE9-specific initialization
        self.spe9_data = None
        self.spe9_metadata = None
        
    def load_spe9_data(self, use_real_data: bool = True) -> Dict[str, Any]:
        """Load SPE9 data (real or synthetic)."""
        from src.data.downloader import SPE9Downloader
        from src.data.preprocessor import SPE9Preprocessor
        
        # Download or use existing data
        downloader = SPE9Downloader()
        
        if use_real_data:
            success = downloader.download(source='opm')
            if not success:
                logger.warning("Failed to download real SPE9 data, using synthetic")
                data_dir = downloader.create_synthetic_if_missing()
            else:
                data_dir = downloader.raw_dir
        else:
            data_dir = downloader.create_synthetic_if_missing()
        
        # Preprocess data
        preprocessor = SPE9Preprocessor(self.config.data_config)
        data_splits = preprocessor.preprocess(data_dir)
        
        # Get dataset info
        dataset_info = preprocessor.get_dataset_info()
        
        self.spe9_data = data_splits
        self.spe9_metadata = dataset_info
        
        logger.info(f"SPE9 Dataset Info: {dataset_info}")
        
        return data_splits
    
    def run_spe9_experiment(self, model_type: str = 'esn') -> Dict[str, Any]:
        """Run experiment on SPE9 dataset."""
        if self.spe9_data is None:
            self.load_spe9_data()
        
        # Prepare data
        X_train = self.spe9_data['X_train']
        y_train = self.spe9_data['y_train']
        X_val = self.spe9_data['X_val']
        y_val = self.spe9_data['y_val']
        
        # Reshape if needed
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train.reshape(y_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_val_flat = y_val.reshape(y_val.shape[0], -1)
        
        # Run experiment based on model type
        if model_type == 'esn':
            from src.models.esn import EchoStateNetwork, ESNConfig
            
            config = ESNConfig(
                n_inputs=X_train_flat.shape[1],
                n_outputs=y_train_flat.shape[1],
                **self.config.esn_config
            )
            
            model = EchoStateNetwork(config)
            stats = model.fit(X_train_flat, y_train_flat, 
                            validation_data=(X_val_flat, y_val_flat))
            
            predictions = model.predict(X_val_flat)
            
        elif model_type == 'physics_informed':
            from src.models.advanced_esn import PhysicsInformedESN, PhysicsInformedESNConfig
            from src.models.esn import ESNConfig
            
            base_config = ESNConfig(
                n_inputs=X_train_flat.shape[1],
                n_outputs=y_train_flat.shape[1],
                **self.config.esn_config
            )
            
            # Define physics constraints for SPE9
            physics_constraints = {
                "material_balance": {
                    "compressibility": 1e-5,
                    "volume": 1e9,  # Reservoir volume
                },
                "boundary_conditions": {
                    "left_value": 3000,
                    "right_value": 3000,
                }
            }
            
            physics_config = PhysicsInformedESNConfig(
                base_config=base_config,
                physics_constraints=physics_constraints,
                constraint_weight=0.1
            )
            
            model = PhysicsInformedESN(physics_config)
            
            # Additional data for physics constraints
            additional_data = self._prepare_physics_data()
            
            stats = model.fit(X_train_flat, y_train_flat, 
                            additional_data=additional_data)
            
            predictions = model.predict(X_val_flat)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate
        from src.utils.metrics import PetroleumMetrics
        metrics = PetroleumMetrics.comprehensive_metrics(y_val_flat, predictions)
        
        results = {
            'model_type': model_type,
            'dataset': 'SPE9',
            'metrics': metrics,
            'stats': stats,
            'model_summary': model.summary() if hasattr(model, 'summary') else str(model),
            'spe9_metadata': self.spe9_metadata,
        }
        
        return results
    
    def _prepare_physics_data(self) -> Dict[str, np.ndarray]:
        """Prepare additional data for physics-informed models."""
        # Extract time from data
        time = np.arange(self.spe9_data['X_train'].shape[0] + 
                        self.spe9_data['X_val'].shape[0])
        
        # For SPE9, we might have pressure and production data
        # This is simplified - in practice, extract from actual data
        return {
            'time': time,
            'pressure': np.random.uniform(3000, 3500, len(time)),
            'production': np.random.uniform(1000, 5000, len(time)),
            'compressibility': 1e-5,
            'volume': 1e9,
        }
