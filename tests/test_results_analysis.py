"""
Test results analysis functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from analysis.performance_calculator import PerformanceCalculator
from analysis.plot_generator import PlotGenerator


class TestPerformanceCalculator:
    """Test performance calculator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample summary data for testing"""
        dates = pd.date_range('2015-01-01', periods=365, freq='D')
        
        # Create realistic production data
        time = np.arange(365)
        oil_rate = 1500 * np.exp(-0.002 * time) + np.random.normal(0, 50, 365)
        gas_rate = oil_rate * 0.8 + np.random.normal(0, 30, 365)
        water_rate = oil_rate * 0.1 * (time/365) + np.random.normal(0, 20, 365)
        
        data = pd.DataFrame({
            'DATE': dates,
            'FOPR': np.maximum(oil_rate, 0),
            'FOPT': np.cumsum(np.maximum(oil_rate, 0)),
            'FGPR': np.maximum(gas_rate, 0),
            'FGPT': np.cumsum(np.maximum(gas_rate, 0)),
            'FWPR': np.maximum(water_rate, 0),
            'FWPT': np.cumsum(np.maximum(water_rate, 0)),
            'FWCT': np.maximum(water_rate, 1) / np.maximum(oil_rate + water_rate, 1),
            'FGOR': np.maximum(gas_rate, 1) / np.maximum(oil_rate, 1),
        })
        
        return data
    
    def test_calculator_initialization(self, sample_data):
        """Test calculator initialization"""
        calculator = PerformanceCalculator(sample_data)
        assert calculator.data is not None
        assert len(calculator.data) == 365
        
    def test_production_metrics(self, sample_data):
        """Test production metrics calculation"""
        calculator = PerformanceCalculator(sample_data)
        metrics = calculator.calculate_all_metrics()
        
        assert 'cumulative_oil' in metrics
        assert 'peak_oil_rate' in metrics
        assert 'avg_oil_rate' in metrics
        assert 'final_oil_rate' in metrics
        
        # Verify calculations
        assert metrics['cumulative_oil'] == pytest.approx(sample_data['FOPT'].max())
        assert metrics['peak_oil_rate'] == pytest.approx(sample_data['FOPR'].max())
        assert metrics['avg_oil_rate'] == pytest.approx(sample_data['FOPR'].mean())
        
    def test_economic_metrics(self, sample_data):
        """Test economic metrics calculation"""
        calculator = PerformanceCalculator(sample_data)
        metrics = calculator.calculate_all_metrics()
        
        economic_metrics = ['gross_revenue', 'net_revenue', 'revenue_per_barrel']
        
        for metric in economic_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.floating))
            
    def test_reservoir_metrics(self, sample_data):
        """Test reservoir metrics calculation"""
        calculator = PerformanceCalculator(sample_data)
        metrics = calculator.calculate_all_metrics()
        
        assert 'oil_recovery_factor' in metrics
        assert 'gas_recovery_factor' in metrics
        assert 'annual_decline_rate' in metrics
        
        # Recovery factors should be percentages
        assert 0 <= metrics['oil_recovery_factor'] <= 100
        assert 0 <= metrics['gas_recovery_factor'] <= 100
        
    def test_detailed_report(self, sample_data):
        """Test detailed report generation"""
        calculator = PerformanceCalculator(sample_data)
        calculator.calculate_all_metrics()
        
        report_df = calculator.generate_detailed_report()
        
        assert isinstance(report_df, pd.DataFrame)
        assert not report_df.empty
        assert 'Value' in report_df.columns
        assert 'Unit' in report_df.columns
        assert 'Description' in report_df.columns
        
        # Verify all metrics are in report
        assert len(report_df) == len(calculator.metrics)
        
    def test_empty_data(self):
        """Test with empty data"""
        empty_data = pd.DataFrame()
        calculator = PerformanceCalculator(empty_data)
        
        metrics = calculator.calculate_all_metrics()
        assert metrics == {}  # Should return empty dict
        
    def test_partial_data(self):
        """Test with partial data"""
        partial_data = pd.DataFrame({
            'DATE': pd.date_range('2015-01-01', periods=10),
            'FOPR': range(10),
            'FOPT': np.cumsum(range(10))
        })
        
        calculator = PerformanceCalculator(partial_data)
        metrics = calculator.calculate_all_metrics()
        
        # Should still calculate available metrics
        assert 'cumulative_oil' in metrics
        assert 'peak_oil_rate' in metrics
        # Economic metrics might be missing or zero


class TestPlotGenerator:
    """Test plot generator"""
    
    @pytest.fixture
    def plot_generator(self, tmp_path):
        """Create plot generator with temp directory"""
        return PlotGenerator(str(tmp_path))
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for plotting"""
        dates = pd.date_range('2015-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'DATE': dates,
            'FOPR': 1500 + np.random.randn(30) * 100,
            'FOPT': np.cumsum(1500 + np.random.randn(30) * 100),
            'FGPR': 1000 + np.random.randn(30) * 50,
            'FGPT': np.cumsum(1000 + np.random.randn(30) * 50),
            'FWPR': 500 + np.random.randn(30) * 25,
            'FWPT': np.cumsum(500 + np.random.randn(30) * 25),
            'FWCT': np.linspace(0, 0.3, 30),
            'FGOR': 500 + np.random.randn(30) * 20,
            'WOPR:PRODU1': 100 + np.random.randn(30) * 10,
            'WOPR:PRODU2': 90 + np.random.randn(30) * 8,
        })
    
    def test_plot_generator_initialization(self, plot_generator, tmp_path):
        """Test plot generator initialization"""
        assert plot_generator.results_dir == Path(tmp_path)
        assert plot_generator.output_dir.exists()
        
    def test_production_profile_plot(self, plot_generator, minimal_data):
        """Test production profile plot generation"""
        plot_path = plot_generator.create_production_profile(minimal_data)
        
        assert plot_path is not None
        assert isinstance(plot_path, str)
        assert plot_path.endswith('.png')
        assert Path(plot_path).exists()
        
        # Clean up
        Path(plot_path).unlink()
        
    def test_water_cut_plot(self, plot_generator, minimal_data):
        """Test water cut plot generation"""
        plot_path = plot_generator.create_water_cut_plot(minimal_data)
        
        assert plot_path is not None
        assert plot_path.endswith('.png')
        assert Path(plot_path).exists()
        
        # Clean up
        Path(plot_path).unlink()
        
    def test_recovery_factor_plot(self, plot_generator, minimal_data):
        """Test recovery factor plot generation"""
        plot_path = plot_generator.create_recovery_factor_plot(minimal_data, ooip=1e6)
        
        assert plot_path is not None
        assert plot_path.endswith('.png')
        assert Path(plot_path).exists()
        
        # Clean up
        Path(plot_path).unlink()
        
    def test_well_performance_chart(self, plot_generator, minimal_data):
        """Test well performance chart generation"""
        plot_path = plot_generator.create_well_performance_chart(minimal_data)
        
        assert plot_path is not None
        assert plot_path.endswith('.png')
        assert Path(plot_path).exists()
        
        # Clean up
        Path(plot_path).unlink()
        
    def test_all_plots_generation(self, plot_generator, minimal_data):
        """Test generation of all plots"""
        plots = plot_generator.create_all_plots(minimal_data)
        
        assert isinstance(plots, dict)
        assert len(plots) > 0
        
        for plot_name, plot_path in plots.items():
            assert plot_name in ['production_profile', 'well_performance', 
                               'water_cut', 'recovery_factor']
            assert Path(plot_path).exists()
            
        # Clean up
        for plot_path in plots.values():
            Path(plot_path).unlink()
            
    def test_missing_data_handling(self, plot_generator):
        """Test handling of missing data"""
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            'DATE': pd.date_range('2015-01-01', periods=10),
            'FOPR': range(10)  # Missing other required columns
        })
        
        # Should handle gracefully
        plot_path = plot_generator.create_production_profile(incomplete_data)
        # Might return empty string or handle gracefully
        assert plot_path is not None


def test_integration(sample_data, tmp_path):
    """Test integration of calculator and plot generator"""
    # Calculate metrics
    calculator = PerformanceCalculator(sample_data)
    metrics = calculator.calculate_all_metrics()
    
    assert len(metrics) > 10  # Should have many metrics
    
    # Generate plots
    generator = PlotGenerator(str(tmp_path))
    plots = generator.create_all_plots(sample_data)
    
    assert len(plots) >= 2  # Should generate at least 2 plots
    
    # Clean up
    for plot_path in plots.values():
        if Path(plot_path).exists():
            Path(plot_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
