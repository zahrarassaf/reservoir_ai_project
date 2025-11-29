import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from config.model_config import Config, ModelFactoryConfig
from src.data_loader import ReservoirDataLoader
from src.feature_engineer import AdvancedFeatureEngineer
from src.model_factory import ReservoirModelFactory
from src.ensemble_trainer import AdvancedEnsembleTrainer
from src.evaluator import ComprehensiveEvaluator

def main():
    print("🚀 RESERVOIR AI - PROFESSIONAL TRAINING PIPELINE")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Data Loading & Preparation
    print("\n📊 STEP 1: DATA LOADING & PREPARATION")
    print("-" * 40)
    
    data_loader = ReservoirDataLoader(config)
    df = data_loader.load_and_validate_data()
    
    # Step 2: Sequence Creation
    X, y, feature_names = data_loader.create_sequences(df)
    
    # Step 3: Train-Test Split (Time Series Aware)
    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split
