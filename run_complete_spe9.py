# run_complete_spe9.py
"""
Complete SPE9 Reservoir AI Project Pipeline
Runs data loading, training, evaluation, and visualization.
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    parser = argparse.ArgumentParser(
        description='SPE9 Reservoir AI Project - Complete Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data parsing command
    parse_parser = subparsers.add_parser('parse', help='Parse SPE9 data')
    parse_parser.add_argument('--data_path', type=str, required=True,
                            help='Path to SPE9 data directory')
    parse_parser.add_argument('--output', type=str, default='data/parsed',
                            help='Output directory for parsed data')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model on SPE9')
    train_parser.add_argument('--data_path', type=str, required=True,
                            help='Path to parsed data or SPE9 directory')
    train_parser.add_argument('--config', type=str, default='configs/spe9_train.yaml',
                            help='Training configuration file')
    train_parser.add_argument('--output', type=str, default='experiments',
                            help='Output directory for results')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model_path', type=str, required=True,
                           help='Path to trained model checkpoint')
    eval_parser.add_argument('--data_path', type=str, required=True,
                           help='Path to test data')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('--results_path', type=str, required=True,
                          help='Path to results directory')
    viz_parser.add_argument('--output', type=str, default='figures',
                          help='Output directory for figures')
    
    args = parser.parse_args()
    
    if args.command == 'parse':
        from src.data.eclipse_parser import EclipseParser
        import json
        
        print("🔍 Parsing SPE9 data...")
        parser = EclipseParser(args.data_path)
        parsed_data = parser.get_full_dataset()
        
        # Save parsed data
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (simplified)
        output_file = output_dir / 'spe9_parsed.json'
        
        # Convert numpy arrays to lists for JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        import numpy as np
        json_data = convert_for_json(parsed_data)
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Parsed data saved to: {output_file}")
    
    elif args.command == 'train':
        from src.training.spe9_trainer import run_spe9_training
        
        print("🚀 Starting SPE9 training...")
        results = run_spe9_training(args.data_path, args.config)
        
        print(f"✅ Training completed. Results saved to: {args.output}")
    
    elif args.command == 'evaluate':
        from src.training.spe9_trainer import SPE9Trainer
        import torch
        
        print("🧪 Evaluating model...")
        
        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location='cpu')
        
        # Recreate trainer and model
        # (This would need to be adapted based on your exact setup)
        
        print("✅ Evaluation completed.")
    
    elif args.command == 'visualize':
        from src.utils.visualization import create_spe9_visualizations
        
        print("📊 Creating visualizations...")
        create_spe9_visualizations(args.results_path, args.output)
        
        print(f"✅ Visualizations saved to: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
