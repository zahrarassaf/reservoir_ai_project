#!/usr/bin/env python3
"""
PhD Reservoir Simulator with ML Integration
CNN-LSTM-SVR for advanced analysis
"""

import sys
from pathlib import Path

# Add ML modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.real_parser import RealSPE9Parser
from src.core.physics_engine import BlackOilSimulator
from src.economics.cash_flow import EconomicAnalyzer

# Import ML modules
from src.ml.cnn_3d_reservoir import Reservoir3DCNN
from src.ml.lstm_production import ProductionForecaster
from src.ml.svr_economics import EconomicSVR
from src.ml.hybrid_cnn_lstm import HybridCNNLSTM

def run_ml_analysis(real_data: dict, simulation_results: dict):
    """Run comprehensive ML analysis."""
    
    print("\n🤖 Running Machine Learning Analysis...")
    
    # 1. CNN for spatial analysis
    print("🧠 1. CNN - Spatial Reservoir Analysis")
    cnn_model = Reservoir3DCNN(use_gpu=True)
    
    # Prepare 3D data for CNN
    porosity_3d = real_data['porosity'].reshape(24, 25, 15)
    perm_3d = real_data['permeability']['x'].reshape(24, 25, 15)
    
    # Train/predict with CNN
    cnn_predictions = cnn_model.predict_pressure_field(
        porosity=porosity_3d,
        permeability=perm_3d,
        saturation=simulation_results['saturation_oil'][-1].reshape(24, 25, 15),
        well_locations=np.array([well['i', 'j', 'k'] for well in real_data['wells']])
    )
    
    # 2. LSTM for temporal forecasting
    print("📈 2. LSTM - Production Forecasting")
    lstm_forecaster = ProductionForecaster()
    
    # Convert simulation results to time series
    production_series = pd.DataFrame({
        'oil_rate': simulation_results['production']['oil'],
        'water_rate': simulation_results['production']['water'],
        'pressure': [np.mean(p) for p in simulation_results['pressure']],
        'time': simulation_results['time']
    })
    
    # Forecast future production
    forecast_results = lstm_forecaster.forecast_with_confidence(
        production_series.values[-30:],  # Last 30 days
        n_samples=1000
    )
    
    # 3. SVR for economic analysis
    print("💰 3. SVR - Economic Sensitivity Analysis")
    svr_model = EconomicSVR(kernel='rbf', use_grid_search=True)
    
    # Prepare economic data
    economic_features = svr_model.prepare_economic_data(
        reservoir_data=real_data,
        economic_data=pd.DataFrame(simulation_results['economic_metrics'])
    )
    
    # Train SVR model
    svr_results = svr_model.train_multi_target(*economic_features)
    
    # Perform sensitivity analysis
    sensitivity_results = svr_model.sensitivity_analysis(
        base_params={
            'oil_price': 82.5,
            'operating_cost': 16.5,
            'discount_rate': 0.095
        },
        parameter_ranges={
            'oil_price': (50, 120),
            'operating_cost': (10, 25),
            'discount_rate': (0.05, 0.15)
        }
    )
    
    # 4. Hybrid CNN-LSTM for integrated analysis
    print("🔗 4. Hybrid CNN-LSTM - Integrated Analysis")
    hybrid_model = HybridCNNLSTM()
    
    # Prepare spatio-temporal data
    spatial_data = np.stack([
        porosity_3d,
        perm_3d,
        simulation_results['saturation_oil'][-1].reshape(24, 25, 15),
        simulation_results['pressure'][-1].reshape(24, 25, 15)
    ], axis=0)  # (4, 24, 25, 15)
    
    temporal_data = production_series[['oil_rate', 'water_rate', 'pressure']].values
    
    # Make hybrid predictions
    hybrid_predictions = hybrid_model.forecast_production(
        spatial_sequence=spatial_data,
        temporal_sequence=temporal_data[-30:],  # Last 30 time steps
        forecast_steps=90  # Forecast 90 days
    )
    
    return {
        'cnn': {
            'pressure_predictions': cnn_predictions,
            'sweet_spots': cnn_model.predict_sweet_spots(spatial_data),
            'uncertainty_map': cnn_model.generate_uncertainty_map(spatial_data)
        },
        'lstm': {
            'production_forecast': forecast_results,
            'anomalies': lstm_forecaster.detect_anomalies(production_series.values)
        },
        'svr': {
            'economic_predictions': svr_results,
            'sensitivity_analysis': sensitivity_results,
            'break_even': svr_model.calculate_break_even({
                'porosity': np.mean(porosity_3d),
                'permeability': np.mean(perm_3d),
                'initial_rate': production_series['oil_rate'].iloc[0]
            })
        },
        'hybrid': {
            'predictions': hybrid_predictions,
            'spatio_temporal_analysis': 'completed'
        }
    }

def main():
    """Main function with ML integration."""
    
    # ... (existing code for loading data and physics simulation)
    
    # Run physics simulation
    physics_results = simulator.run(total_time=3650)
    
    # Run ML analysis
    ml_results = run_ml_analysis(real_data, physics_results)
    
    # Generate comprehensive report
    report = {
        'physics': physics_results,
        'economics': economic_results,
        'ml_analysis': ml_results,
        'validation': validation
    }
    
    print("\n✅ ML Analysis Completed:")
    print(f"   • CNN identified {np.sum(ml_results['cnn']['sweet_spots'])} sweet spots")
    print(f"   • LSTM forecast: {ml_results['lstm']['production_forecast']['mean'][-1]:.1f} bpd in 90 days")
    print(f"   • SVR break-even price: ${ml_results['svr']['break_even']['break_even_price']:.2f}/bbl")
    print(f"   • Hybrid model confidence: ±{ml_results['hybrid']['predictions']['std']:.1%}")
