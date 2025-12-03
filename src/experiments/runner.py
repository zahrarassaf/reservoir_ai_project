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
        if model
