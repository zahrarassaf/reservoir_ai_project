# run_real_spe9.py
"""
Run complete pipeline with YOUR real SPE9 data.
"""

import argparse
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Run REAL SPE9 pipeline')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to your SPE9 data directory or archive')
    parser.add_argument('--output', type=str, default='real_spe9_results',
                       help='Output directory')
    parser.add_argument('--train', action='store_true',
                       help='Train model on real data')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize data')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 REAL SPE9 PIPELINE - YOUR DATA")
    print("="*70)
    
    # Step 1: Analyze your data
    from scripts.analyze_real_spe9 import analyze_spe9_archive
    print(f"\n1. 📦 ANALYZING YOUR DATA: {args.data_path}")
    analysis = analyze_spe9_archive(args.data_path)
    
    # Step 2: Load data
    print(f"\n2. 📂 LOADING REAL DATA")
    from src.data.real_spe9_loader import RealSPE9Loader
    
    loader = RealSPE9Loader('data/spe9_raw')
    all_data = loader.load_all()
    
    # Step 3: Get training data
    print(f"\n3. 🎯 PREPARING TRAINING DATA")
    X, y = loader.get_training_data(sequence_length=10)
    
    print(f"   Training samples: {len(X)}")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Step 4: Train if requested
    if args.train:
        print(f"\n4. 🧠 TRAINING MODEL")
        
        # Import and train
        from src.training.spe9_trainer import SPE9Trainer
        from src.models.reservoir_neural_operator import ReservoirNeuralOperator
        
        # Create dataset class
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create model
        grid_info = loader.get_grid_info()
        model = ReservoirNeuralOperator(
            input_channels=2,
            output_channels=2,
            grid_dims=grid_info['dims'],
            hidden_channels=64,
            num_fno_layers=4,
            fno_modes=12
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"   Model on: {device}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Create trainer
        from src.training.spe9_trainer import SPE9Trainer
        
        # Simplified training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Training loop
        epochs = 10  # Start with few epochs
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass (simplified)
                predictions, _ = model(batch_X[:, -1])  # Last timestep
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    predictions, _ = model(batch_X[:, -1])
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            
            print(f"   Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    
    # Step 5: Visualize if requested
    if args.visualize:
        print(f"\n5. 📊 VISUALIZING DATA")
        loader.visualize_data(args.output)
    
    print(f"\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print(f"   Data analyzed: {analysis.get('total_files', 0)} files")
    print(f"   Grid: {loader.grid_info.get('dims', 'Unknown')}")
    print(f"   Output directory: {args.output}")
    print("="*70)

if __name__ == "__main__":
    main()
