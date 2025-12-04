# run_phd_experiment.py - MAIN EXECUTION
import torch
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def check_environment():
    """Check PhD-level environment requirements."""
    print("🔬 PHD-LEVEL ENVIRONMENT CHECK")
    print("="*50)
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check dependencies
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except:
        print("❌ NumPy not installed")
    
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except:
        print("⚠️ SciPy not installed (needed for statistical tests)")
    
    try:
        import wandb
        print("✅ Weights & Biases available")
    except:
        print("⚠️ wandb not installed (logging disabled)")
    
    print("="*50)
    return True

def run_phd_experiment():
    """Run complete PhD-level experiment."""
    from src.experiments.runner import run_experiment, ExperimentConfig
    
    print("\n🚀 STARTING PHD-LEVEL EXPERIMENT")
    print("="*60)
    
    # Configuration
    config = ExperimentConfig(
        data_path="data/spe9",
        model_type="physics_informed_esn",
        reservoir_size=2000,
        physics_weight=0.15,
        epochs=50,  # Start with 50 for testing
        batch_size=4,
        output_dir="results/phd_experiment_1"
    )
    
    # Run experiment
    results = run_experiment()
    
    print("\n✅ EXPERIMENT COMPLETE")
    print(f"Results saved to: {config.output_dir}")
    
    return results

def main():
    """Main entry point."""
    # Check environment
    check_environment()
    
    # Run experiment
    try:
        results = run_phd_experiment()
        
        # Print summary
        if 'evaluation' in results:
            eval_res = results['evaluation']
            if 'statistical_tests' in eval_res:
                stats = eval_res['statistical_tests']
                print(f"\n📊 FINAL RESULTS:")
                print(f"   Test Loss: {eval_res.get('test_loss', 0):.6f}")
                print(f"   R² Score: {stats.get('r2_score', 0):.4f}")
                print(f"   RMSE: {stats.get('rmse', 0):.4f}")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple test
        print("\n🔄 FALLBACK: Running minimal test...")
        run_minimal_test()

def run_minimal_test():
    """Minimal test if main experiment fails."""
    print("\n🧪 MINIMAL TEST MODE")
    
    # Create synthetic data
    import numpy as np
    import torch.nn as nn
    
    # Simple data
    n_samples = 100
    n_features = 10
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 2)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(n_features, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("✅ Minimal test completed")

if __name__ == "__main__":
    main()
