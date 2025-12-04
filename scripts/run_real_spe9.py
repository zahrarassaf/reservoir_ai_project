#!/usr/bin/env python3
"""
Run experiments with real SPE9 data.
"""

import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.real_spe9_parser import RealSPE9Parser
from src.data.downloader import SPE9Downloader
from src.experiments.runner import SPE9ExperimentRunner
from src.utils.advanced_logger import get_logger, LoggingConfig

# Configure logging
logging_config = LoggingConfig(
    console_level="INFO",
    file_level="DEBUG",
    log_dir=Path("logs/spe9_real"),
)

logger = get_logger(logging_config)

def main():
    """Main function to run real SPE9 experiments."""
    
    # Download real SPE9 data
    downloader = SPE9Downloader()
    
    print("📥 Downloading real SPE9 data...")
    success = downloader.download(source='opm')
    
    if not success:
        print("⚠️ Could not download real data, using synthetic...")
        data_dir = downloader.create_synthetic_if_missing()
    else:
        data_dir = downloader.raw_dir
    
    print(f"📁 Data directory: {data_dir}")
    
    # Parse real SPE9 data
    parser = RealSPE9Parser(data_dir)
    
    print("🔍 Parsing SPE9 files...")
    dataset = parser.get_complete_dataset()
    
    # Print dataset info
    metadata = dataset.get('metadata', {})
    print("\n📊 SPE9 Dataset Information:")
    print(f"  Name: {metadata.get('name')}")
    print(f"  Grid: {metadata.get('grid')}")
    print(f"  Wells: {metadata.get('wells')}")
    print(f"  Period: {metadata.get('simulation_period')}")
    
    if 'summary' in dataset and dataset['summary'] is not None:
        summary_df = dataset['summary']
        print(f"\n📈 Summary Data: {summary_df.shape}")
        print(f"  Time steps: {len(summary_df)}")
        print(f"  Variables: {len(summary_df.columns)}")
        print(f"  Available: {list(summary_df.columns)[:10]}...")
    
    if 'grid' in dataset and dataset['grid'] is not None:
        grid_df = dataset['grid']
        print(f"\n🗺️  Grid Data: {grid_df.shape}")
        print(f"  Active cells: {grid_df['ACTIVE'].sum() if 'ACTIVE' in grid_df.columns else 'N/A'}")
    
    # Save dataset for inspection
    output_dir = Path("output/spe9_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'summary' in dataset and dataset['summary'] is not None:
        summary_df.to_csv(output_dir / "spe9_summary.csv", index=False)
        print(f"\n💾 Saved summary to: {output_dir / 'spe9_summary.csv'}")
    
    # Run experiment
    print("\n🧪 Running reservoir computing experiment on SPE9 data...")
    
    from src.models.esn import EchoStateNetwork, ESNConfig
    from src.utils.metrics import PetroleumMetrics
    
    # Prepare data from summary
    summary_df = dataset['summary']
    
    # Use oil rate as target, other variables as features
    target_col = 'FOPR' if 'FOPR' in summary_df.columns else summary_df.columns[1]
    feature_cols = [col for col in summary_df.columns 
                   if col not in ['TIME', target_col]][:10]  # Use first 10 features
    
    print(f"  Target: {target_col}")
    print(f"  Features: {feature_cols}")
    
    X = summary_df[feature_cols].values
    y = summary_df[[target_col]].values
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Configure ESN
    config = ESNConfig(
        n_inputs=X_train.shape[1],
        n_outputs=y_train.shape[1],
        n_reservoir=500,
        spectral_radius=0.95,
        sparsity=0.1,
        leaking_rate=0.3,
        regularization=1e-6,
        random_state=42
    )
    
    # Train model
    print("  Training ESN...")
    esn = EchoStateNetwork(config)
    stats = esn.fit(X_train, y_train)
    
    # Make predictions
    print("  Making predictions...")
    y_pred = esn.predict(X_test)
    
    # Evaluate
    metrics = PetroleumMetrics.comprehensive_metrics(y_test, y_pred)
    
    print("\n📊 Results on SPE9 Data:")
    print(f"  Nash-Sutcliffe Efficiency: {metrics.get('nash_sutcliffe', 0):.4f}")
    print(f"  R² Score: {metrics.get('r2', 0):.4f}")
    print(f"  RMSE: {metrics.get('rmse', 0):.2f}")
    
    # Save results
    results = {
        'dataset': 'SPE9_Real',
        'target_variable': target_col,
        'features': feature_cols,
        'model_config': config.__dict__,
        'metrics': metrics,
        'data_shape': {
            'train': X_train.shape,
            'test': X_test.shape,
        }
    }
    
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Real SPE9 experiment completed successfully!")

if __name__ == "__main__":
    main()
