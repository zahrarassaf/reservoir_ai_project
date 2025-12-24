"""
RESERVOIR AI PROJECT - SPE9 DATA ANALYSIS
Integrated Physics-Based Simulation with Machine Learning
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project paths
sys.path.append('src')
sys.path.append('models')
sys.path.append('data_processing')

# Import modules
from data_loader import SPE9DataLoader
from reservoir_simulator import ReservoirSimulator
from cnn_property_predictor import PropertyPredictor
from economic_analyzer import EconomicAnalyzer
from visualization import ReservoirVisualizer
from report_generator import ReportGenerator

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("RESERVOIR SIMULATION - SPE9 DATA ANALYSIS")
    print("=" * 70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load SPE9 data
    print("\nLoading SPE9 datasets...")
    data_loader = SPE9DataLoader('data/spe9')
    datasets = data_loader.load_all_datasets()
    
    if not datasets:
        print("ERROR: No data files found!")
        return
    
    print(f"Found {len(datasets)} data files:")
    for filename, (data, size_kb) in datasets.items():
        print(f"   {filename:30s} {size_kb:7.1f} KB")
    
    # Step 2: Parse grid data
    print("\nParsing SPE9.GRDECL (grid data)...")
    grid_data = data_loader.parse_grid_data()
    if grid_data:
        nx, ny, nz = grid_data['dims']
        print(f"   Grid: ({nx}, {ny}, {nz}) = {nx*ny*nz:,} cells")
    
    # Step 3: Parse permeability and tops
    print("Parsing PERMVALUES.DATA...")
    perm_data = data_loader.parse_permeability()
    if perm_data is not None:
        print(f"   Permeability: {len(perm_data)} values loaded")
        
        # Display permeability statistics
        print("\n=== PERMEABILITY DATA VALIDATION ===")
        print(f"Total values: {len(perm_data)}")
        print(f"Value range: {perm_data.min():.1f} to {perm_data.max():.1f} md")
        print(f"Mean permeability: {perm_data.mean():.1f} md")
        print(f"Std deviation: {perm_data.std():.1f} md")
        print(f"First 10 values: {perm_data[:10]}")
    
    print("Parsing TOPSVALUES.DATA...")
    tops_data = data_loader.parse_tops()
    if tops_data is not None:
        print(f"   Tops: {len(tops_data)} values loaded")
    
    # Step 4: Parse main SPE9 configuration
    print("Parsing SPE9.DATA...")
    spe9_config = data_loader.parse_spe9_config()
    if spe9_config:
        print(f"   SPE9 Configuration: {spe9_config.get('grid', 'N/A')}")
    
    # Step 5: Find SPE9 variants
    print("\nFound SPE9 variants:")
    variants = data_loader.find_spe9_variants()
    for variant in variants:
        print(f"   {variant}")
    
    # Step 6: Setup and run reservoir simulation
    print("\nSetting up reservoir from data...")
    
    # Match data sizes
    if perm_data is not None and grid_data is not None:
        total_cells = nx * ny * nz
        if len(perm_data) != total_cells:
            print(f"Warning: Permeability array size ({len(perm_data)}) doesn't match grid ({total_cells})")
            if len(perm_data) > total_cells:
                print(f"Truncating to {total_cells} values")
                perm_data = perm_data[:total_cells]
            else:
                print(f"Padding with mean value")
                mean_perm = perm_data.mean()
                perm_data = np.pad(perm_data, (0, total_cells - len(perm_data)), 
                                 'constant', constant_values=mean_perm)
    
    # Create reservoir
    reservoir = ReservoirSimulator(
        grid_dims=(nx, ny, nz),
        permeability=perm_data,
        porosity=0.2,  # Default value
        initial_pressure=5000,  # psi
        initial_saturation=0.8
    )
    
    print("Reservoir setup complete:")
    print(f"Grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} cells")
    print(f"Permeability: {perm_data.mean():.1f} ± {perm_data.std():.1f} md")
    
    # Data validation display
    print("\n=== DATA VALIDATION ===")
    print(f"Permeability values loaded: {len(perm_data)}")
    print(f"Permeability range: {perm_data.min():.1f} to {perm_data.max():.1f} md")
    print(f"First 5 permeability values: {perm_data[:5]}")
    
    # Step 7: Run physics-based simulation
    print("\nRunning physics-based simulation for 10 years...")
    simulation_results = reservoir.run_simulation(years=10)
    
    # Step 8: Calculate well productivity
    print("\nCalculating well productivity...")
    production = reservoir.calculate_production()
    
    print(f"Initial production rate: {production['initial_rate']:.0f} bpd")
    print(f"Oil in place: {production['oil_in_place']:.1f} MM bbl")
    print(f"Recoverable oil: {production['recoverable_oil']:.1f} MM bbl")
    
    # Step 9: Economic analysis
    print("\nRunning economic analysis...")
    economic_results = reservoir.run_economic_analysis(
        oil_price=60,
        operating_cost=20,
        capital_cost=17500000
    )
    
    print("\n" + "=" * 70)
    print("MACHINE LEARNING INTEGRATION")
    print("=" * 70)
    
    # Step 10: CNN Property Prediction
    print("\nRunning CNN property prediction...")
    
    # Prepare data for CNN
    print(f"Input grid shape: {perm_data.reshape(nx, ny, nz).shape}")
    print(f"Input grid type: {type(perm_data)}")
    
    # Create and train CNN model
    cnn_model = PropertyPredictor(
        input_shape=(nx, ny, nz),
        learning_rate=0.001,
        epochs=10
    )
    
    # Train model (generate synthetic data)
    grid_3d = perm_data.reshape(nx, ny, nz)
    print(f"Grid data shape: {grid_3d.shape}")
    
    # Generate training and test data
    n_samples = 100
    X_train = []
    y_train = []
    
    for i in range(n_samples):
        # Create small variations in the main grid
        noise = np.random.normal(0, 0.1, grid_3d.shape)
        sample = grid_3d + noise * grid_3d
        X_train.append(sample)
        
        # Target: predict three parameters (mean, variance, skewness)
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        skew_val = np.mean((sample - mean_val)**3) / (std_val**3 + 1e-6)
        y_train.append([mean_val, std_val, skew_val])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"Training data shape: {X_train_split.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train model
    history = cnn_model.train(
        X_train_split, y_train_split,
        X_val, y_val,
        batch_size=4
    )
    
    # Evaluate model
    test_pred = cnn_model.predict(X_val)
    
    # Calculate performance metrics
    mse = np.mean((test_pred - y_val) ** 2)
    mae = np.mean(np.abs(test_pred - y_val))
    
    # Calculate R² score
    ss_res = np.sum((y_val - test_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val, axis=0)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))
    
    print(f"\nCNN Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save model
    try:
        cnn_model.save_model('results/cnn_reservoir_model.pth')
        print("Model saved to results/cnn_reservoir_model.pth")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    # Step 11: Economic Forecasting with Improved Model
    print("\nRunning economic forecasting...")
    
    economic_model = EconomicAnalyzer(
        model_type='RANDOM_FOREST',
        n_estimators=200,
        random_state=42
    )
    
    # Generate more realistic training data
    print("Training improved economic models...")
    
    # Generate features with more realistic distribution (based on SPE9)
    n_samples_econ = 1000
    features = []
    targets_npv = []
    targets_irr = []
    targets_roi = []
    targets_payback = []
    
    for i in range(n_samples_econ):
        # More realistic features based on SPE9 statistics
        perm_mean = np.random.lognormal(mean=np.log(105), sigma=0.8)
        perm_std = np.random.uniform(50, 400)
        porosity = np.random.normal(0.2, 0.06)
        porosity = np.clip(porosity, 0.1, 0.3)
        thickness = np.random.uniform(50, 200)
        area = np.random.uniform(100, 1000)
        oil_price = np.random.uniform(40, 100)
        op_cost = np.random.uniform(15, 30)
        cap_cost = np.random.uniform(10e6, 30e6)
        
        # Calculate more realistic NPV
        # Simplified reservoir engineering formula
        reservoir_quality = (perm_mean / 100) * porosity * thickness
        recoverable_oil = area * thickness * porosity * 0.8 * 7758  # bbl
        revenue = recoverable_oil * oil_price * 0.5  # 50% recovery factor
        operating_cost_total = recoverable_oil * op_cost * 0.3
        npv = revenue - operating_cost_total - cap_cost
        
        # Ensure realistic ranges
        npv = max(npv, -cap_cost * 0.5)  # Don't lose more than 50% of capital
        npv = min(npv, cap_cost * 5)     # Maximum 5x return
        
        # Calculate other metrics
        irr = np.random.uniform(5, 25) if npv > 0 else np.random.uniform(-10, 5)
        roi = (npv / cap_cost) * 100
        payback = cap_cost / (revenue * 0.1) if revenue > 0 else 20
        
        # Create feature vector
        feature_vector = [
            perm_mean, perm_std, porosity, thickness, area,
            oil_price, op_cost, cap_cost/1e6,
            reservoir_quality, recoverable_oil/1e6, revenue/1e6
        ]
        
        features.append(feature_vector)
        targets_npv.append(npv/1e6)  # Convert to million dollars
        targets_irr.append(irr)
        targets_roi.append(roi)
        targets_payback.append(payback)
    
    features = np.array(features)
    targets_npv = np.array(targets_npv)
    targets_irr = np.array(targets_irr)
    targets_roi = np.array(targets_roi)
    targets_payback = np.array(targets_payback)
    
    print(f"Features: {features.shape[1]}, Samples: {features.shape[0]}")
    
    # Split data for training
    split_idx = int(0.8 * n_samples_econ)
    X_train_econ = features[:split_idx]
    X_test_econ = features[split_idx:]
    
    # Train models for each economic metric
    print("\nTraining models for NPV, IRR, ROI, Payback...")
    
    # Train NPV model
    y_train_npv = targets_npv[:split_idx]
    y_test_npv = targets_npv[split_idx:]
    
    npv_model = economic_model.train_model(
        X_train_econ, y_train_npv,
        model_name='NPV'
    )
    
    # Train IRR model
    y_train_irr = targets_irr[:split_idx]
    y_test_irr = targets_irr[split_idx:]
    
    irr_model = economic_model.train_model(
        X_train_econ, y_train_irr,
        model_name='IRR'
    )
    
    # Train ROI model
    y_train_roi = targets_roi[:split_idx]
    y_test_roi = targets_roi[split_idx:]
    
    roi_model = economic_model.train_model(
        X_train_econ, y_train_roi,
        model_name='ROI'
    )
    
    # Train Payback model
    y_train_payback = targets_payback[:split_idx]
    y_test_payback = targets_payback[split_idx:]
    
    payback_model = economic_model.train_model(
        X_train_econ, y_train_payback,
        model_name='PAYBACK'
    )
    
    # Evaluate models
    print("\nModel Performance:")
    
    # NPV evaluation
    npv_pred = npv_model.predict(X_test_econ)
    npv_mse = np.mean((npv_pred - y_test_npv) ** 2)
    npv_r2 = 1 - npv_mse / np.var(y_test_npv)
    print("NPV:")
    print(f"  MSE: {npv_mse:.4f}")
    print(f"  R²: {npv_r2:.4f}")
    
    # IRR evaluation
    irr_pred = irr_model.predict(X_test_econ)
    irr_mse = np.mean((irr_pred - y_test_irr) ** 2)
    irr_r2 = 1 - irr_mse / np.var(y_test_irr)
    print("IRR:")
    print(f"  MSE: {irr_mse:.4f}")
    print(f"  R²: {irr_r2:.4f}")
    
    # ROI evaluation
    roi_pred = roi_model.predict(X_test_econ)
    roi_mse = np.mean((roi_pred - y_test_roi) ** 2)
    roi_r2 = 1 - roi_mse / np.var(y_test_roi)
    print("ROI:")
    print(f"  MSE: {roi_mse:.4f}")
    print(f"  R²: {roi_r2:.4f}")
    
    # Payback evaluation
    payback_pred = payback_model.predict(X_test_econ)
    payback_mse = np.mean((payback_pred - y_test_payback) ** 2)
    payback_r2 = 1 - payback_mse / np.var(y_test_payback)
    print("Payback:")
    print(f"  MSE: {payback_mse:.4f}")
    print(f"  R²: {payback_r2:.4f}")
    
    # Predict for current SPE9 case
    print("\nEconomic predictions for current case:")
    
    # Extract features from current reservoir
    current_features = [
        perm_data.mean(), perm_data.std(), 0.2,  # porosity
        100,  # thickness (assumed)
        500,  # area (assumed)
        60,   # oil price
        20,   # operating cost
        17.5, # capital cost in millions
        (perm_data.mean()/100) * 0.2 * 100,  # reservoir quality
        production['recoverable_oil'],  # recoverable oil in MMbbl
        production['recoverable_oil'] * 60 * 0.5  # revenue in million
    ]
    
    current_features = np.array(current_features).reshape(1, -1)
    
    # Make predictions
    npv_pred_current = npv_model.predict(current_features)[0]
    irr_pred_current = irr_model.predict(current_features)[0]
    roi_pred_current = roi_model.predict(current_features)[0]
    payback_pred_current = payback_model.predict(current_features)[0]
    
    print(f"npv: {npv_pred_current:.2f} million")
    print(f"irr: {irr_pred_current:.2f}%")
    print(f"roi: {roi_pred_current:.2f}%")
    print(f"payback_period: {payback_pred_current:.2f} years")
    
    # Save economic model
    try:
        economic_model.save_model('results/economic_model.joblib')
        print("Model saved to results/economic_model.joblib")
    except Exception as e:
        print(f"Could not save economic model: {e}")
    
    # Step 12: Generate visualizations
    print("\nGenerating visualizations...")
    
    visualizer = ReservoirVisualizer(reservoir)
    
    # Create comprehensive visualization
    fig = visualizer.create_comprehensive_plot(
        simulation_results=simulation_results,
        production_data=production,
        economic_data=economic_results,
        cnn_performance={'mse': mse, 'mae': mae, 'r2': r2},
        economic_predictions={
            'npv': npv_pred_current,
            'irr': irr_pred_current,
            'roi': roi_pred_current,
            'payback': payback_pred_current
        }
    )
    
    # Save visualization
    fig.savefig('results/spe9_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved: results/spe9_analysis.png")
    
    # Step 13: Generate comprehensive report
    print("\nSaving comprehensive report...")
    
    report_gen = ReportGenerator()
    
    # Compile all results
    full_report = {
        'technical_analysis': {
            'data_source': 'SPE9 Dataset',
            'grid_dimensions': f'{nx}x{ny}x{nz}',
            'total_cells': int(nx * ny * nz),
            'simulation_period': '10 years',
            'peak_production': float(production['peak_rate']),
            'total_oil_recovered': float(production['total_recovered']),
            'average_water_cut': float(simulation_results.get('avg_water_cut', 0)),
            'wells_analyzed': reservoir.n_wells
        },
        'economic_results': {
            'net_present_value': float(economic_results.get('npv', 0)),
            'internal_rate_of_return': float(economic_results.get('irr', 0)),
            'return_on_investment': float(economic_results.get('roi', 0)),
            'payback_period': float(economic_results.get('payback', 0)),
            'break_even_price': float(economic_results.get('break_even', 0)),
            'capital_investment': float(economic_results.get('capital_cost', 0))
        },
        'ml_results': {
            'cnn_property_prediction': {
                'implemented': True,
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2)
            },
            'economic_forecasting': {
                'implemented': True,
                'model_type': 'RANDOM_FOREST',
                'npv_r2': float(npv_r2),
                'irr_r2': float(irr_r2),
                'roi_r2': float(roi_r2),
                'payback_r2': float(payback_r2)
            }
        },
        'data_validation': {
            'data_files_loaded': len(datasets),
            'spe9_variants': len(variants),
            'grid_data_available': grid_data is not None,
            'permeability_data_available': perm_data is not None,
            'permeability_stats': {
                'mean': float(perm_data.mean()),
                'std': float(perm_data.std()),
                'min': float(perm_data.min()),
                'max': float(perm_data.max())
            }
        },
        'output_files': {
            'visualizations': 'results/spe9_analysis.png',
            'json_report': 'results/spe9_report.json',
            'cnn_model': 'results/cnn_reservoir_model.pth',
            'economic_model': 'results/economic_model.joblib'
        },
        'metadata': {
            'execution_date': datetime.now().isoformat(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__ if 'pd' in locals() else 'N/A'
        }
    }
    
    # Save report
    report_path = report_gen.save_report(full_report, 'results/spe9_report.json')
    print(f"Comprehensive report saved: {report_path}")
    
    # Step 14: Display final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)
    
    print("\n    TECHNICAL ANALYSIS:")
    print("    " + "=" * 40)
    print(f"    Data Source: SPE9 Dataset")
    print(f"    Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} cells")
    print(f"    Simulation: 10 years physics-based simulation")
    print(f"    Peak Production: {production['peak_rate']:.0f} bpd")
    print(f"    Total Oil Recovered: {production['total_recovered']:.2f} MM bbl")
    print(f"    Avg Water Cut: {simulation_results.get('avg_water_cut', 0):.1f}%")
    print(f"    Wells Analyzed: {reservoir.n_wells} wells")
    
    print("\n    ECONOMIC RESULTS:")
    print("    " + "=" * 40)
    print(f"    Net Present Value: ${economic_results.get('npv', 0):.2f} Million")
    print(f"    Internal Rate of Return: {economic_results.get('irr', 0):.1f}%")
    print(f"    Return on Investment: {economic_results.get('roi', 0):.1f}%")
    print(f"    Payback Period: {economic_results.get('payback', 0):.1f} years")
    print(f"    Break-even Price: ${economic_results.get('break_even', 0):.1f}/bbl")
    print(f"    Capital Investment: ${economic_results.get('capital_cost', 0)/1e6:.1f} Million")
    
    print("\n    MACHINE LEARNING RESULTS:")
    print("    " + "=" * 40)
    print(f"    CNN Property Prediction: Implemented")
    print(f"    Random Forest Economic Forecasting: Implemented")
    print(f"    CNN Model Accuracy (R²): {r2:.3f}")
    print(f"    Economic Model R² Scores:")
    print(f"      - NPV: {npv_r2:.3f}")
    print(f"      - IRR: {irr_r2:.3f}")
    print(f"      - ROI: {roi_r2:.3f}")
    
    print("\n    DATA VALIDATION:")
    print("    " + "=" * 40)
    print(f"    Data Files: {len(datasets)} files loaded")
    print(f"    SPE9 Variants: {len(variants)} configurations")
    print(f"    Grid Data: {'Available' if grid_data else 'Not Available'}")
    print(f"    Permeability Data: {'Available' if perm_data is not None else 'Not Available'}")
    
    print("\n    OUTPUT FILES:")
    print("    " + "=" * 40)
    print(f"    1. results/spe9_analysis.png - Visualizations")
    print(f"    2. results/spe9_report.json - JSON report")
    print(f"    3. results/cnn_reservoir_model.pth - CNN model")
    print(f"    4. results/economic_model.joblib - Economic model")
    
    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
