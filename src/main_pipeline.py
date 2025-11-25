"""
Main Pipeline Module - Reservoir AI Project
Orchestrates the complete machine learning workflow
"""
import pandas as pd
import numpy as np
import warnings
import time
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import config
from data_preprocessing import DataPreprocessor
from feature_engineering import engineer_features_for_target
from model_training import run_complete_training
from model_evaluation import evaluate_all_models
from shap_analysis import perform_complete_shap_analysis

class ReservoirAIPipeline:
    """
    Main pipeline for Reservoir AI Project
    Coordinates all stages from data loading to model interpretation
    """
    
    def __init__(self, enable_feature_engineering=True, enable_shap=True):
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_shap = enable_shap
        self.pipeline_results = {}
        self.execution_times = {}
        
    def log_step(self, step_name, action):
        """Log pipeline steps with timestamps"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] 🔄 {step_name}: {action}")
        
    def run_complete_pipeline(self):
        """Execute the complete Reservoir AI pipeline"""
        
        print("🚀 STARTING RESERVOIR AI PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Data Preprocessing
            self.log_step("STEP 1", "Data Preprocessing")
            step_time = time.time()
            
            preprocessor = DataPreprocessor(use_robust_scaling=True)
            datasets = preprocessor.prepare_datasets()
            
            if datasets is None:
                raise ValueError("Data preprocessing failed - check your data file")
            
            self.execution_times['data_preprocessing'] = time.time() - step_time
            self.pipeline_results['datasets'] = datasets
            self.pipeline_results['preprocessor'] = preprocessor
            
            print(f"✅ Data preprocessing completed in {self.execution_times['data_preprocessing']:.2f}s")
            
            # Step 2: Feature Engineering
            if self.enable_feature_engineering:
                self.log_step("STEP 2", "Feature Engineering")
                step_time = time.time()
                
                engineered_datasets = {}
                feature_engineers = {}
                
                for target_name in datasets.keys():
                    print(f"\n🎯 Engineering features for: {target_name}")
                    engineered_datasets[target_name], feature_engineer = engineer_features_for_target(
                        datasets, target_name
                    )
                    feature_engineers[target_name] = feature_engineer
                
                self.pipeline_results['datasets'] = engineered_datasets
                self.pipeline_results['feature_engineers'] = feature_engineers
                self.execution_times['feature_engineering'] = time.time() - step_time
                
                print(f"✅ Feature engineering completed in {self.execution_times['feature_engineering']:.2f}s")
            
            # Step 3: Model Training
            self.log_step("STEP 3", "Model Training")
            step_time = time.time()
            
            training_results, model_trainer = run_complete_training(
                self.pipeline_results['datasets']
            )
            
            self.pipeline_results['training_results'] = training_results
            self.pipeline_results['model_trainer'] = model_trainer
            self.execution_times['model_training'] = time.time() - step_time
            
            print(f"✅ Model training completed in {self.execution_times['model_training']:.2f}s")
            
            # Step 4: Model Evaluation
            self.log_step("STEP 4", "Model Evaluation")
            step_time = time.time()
            
            evaluation_reports, model_evaluator = evaluate_all_models(
                training_results, self.pipeline_results['datasets']
            )
            
            self.pipeline_results['evaluation_reports'] = evaluation_reports
            self.pipeline_results['model_evaluator'] = model_evaluator
            self.execution_times['model_evaluation'] = time.time() - step_time
            
            print(f"✅ Model evaluation completed in {self.execution_times['model_evaluation']:.2f}s")
            
            # Step 5: SHAP Analysis
            if self.enable_shap:
                self.log_step("STEP 5", "SHAP Analysis")
                step_time = time.time()
                
                shap_reports, shap_analyzer = perform_complete_shap_analysis(
                    training_results, self.pipeline_results['datasets']
                )
                
                self.pipeline_results['shap_reports'] = shap_reports
                self.pipeline_results['shap_analyzer'] = shap_analyzer
                self.execution_times['shap_analysis'] = time.time() - step_time
                
                print(f"✅ SHAP analysis completed in {self.execution_times['shap_analysis']:.2f}s")
            
            # Final Summary
            total_time = time.time() - start_time
            self.execution_times['total'] = total_time
            
            self._generate_final_summary()
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {total_time:.2f}s")
            
            return self.pipeline_results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise
    
    def _generate_final_summary(self):
        """Generate comprehensive pipeline summary"""
        
        print("\n" + "=" * 60)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        # Execution times
        print("\n⏱️  EXECUTION TIMES:")
        for step, duration in self.execution_times.items():
            print(f"   • {step.replace('_', ' ').title()}: {duration:.2f}s")
        
        # Model performance summary
        print("\n🏆 MODEL PERFORMANCE SUMMARY:")
        
        for target_name, reports in self.pipeline_results['evaluation_reports'].items():
            best_model = reports['best_model']
            print(f"\n   🎯 {target_name}:")
            print(f"      Best Model: {best_model['name']}")
            print(f"      Test R²: {best_model['performance']['Test_R2']:.4f}")
            print(f"      Test RMSE: {best_model['performance']['Test_RMSE']:.4f}")
            print(f"      Test MAE: {best_model['performance']['Test_MAE']:.4f}")
        
        # Feature importance insights
        print("\n🔍 KEY INSIGHTS:")
        
        if 'shap_reports' in self.pipeline_results:
            for target_name, shap_report in self.pipeline_results['shap_reports'].items():
                if 'feature_importance' in shap_report:
                    top_features = shap_report['feature_importance']['dataframe'].head(3)
                    print(f"\n   📈 {target_name} - Top 3 Features:")
                    for idx, row in top_features.iterrows():
                        print(f"      {row['feature']}: {row['importance_percentage']:.1f}%")
        
        # Data statistics
        print(f"\n📈 DATA STATISTICS:")
        preprocessor = self.pipeline_results['preprocessor']
        quality_report = preprocessor.data_quality_report
        
        if quality_report:
            print(f"   • Original samples: {quality_report['original_shape'][0]}")
            print(f"   • Features: {quality_report['original_shape'][1]}")
            print(f"   • Data completeness: {quality_report['quality_metrics']['completeness_score']:.1f}%")
        
        print("\n💾 RESULTS SAVED TO:")
        print(f"   • Models: {config.MODELS_DIR}")
        print(f"   • Results: {config.RESULTS_DIR}")
        print(f"   • Visualizations: {config.RESULTS_DIR}/*.html")
    
    def get_best_models(self):
        """Get the best performing models for each target"""
        
        best_models = {}
        
        if 'evaluation_reports' in self.pipeline_results:
            for target_name, report in self.pipeline_results['evaluation_reports'].items():
                best_model_info = report['best_model']
                best_models[target_name] = {
                    'name': best_model_info['name'],
                    'performance': best_model_info['performance'],
                    'model': self.pipeline_results['training_results'][target_name][best_model_info['name']]['model']
                }
        
        return best_models
    
    def predict_new_data(self, new_data):
        """Make predictions on new data using the best models"""
        
        if 'preprocessor' not in self.pipeline_results:
            raise ValueError("Pipeline not executed yet. Run run_complete_pipeline() first.")
        
        best_models = self.get_best_models()
        predictions = {}
        
        for target_name, model_info in best_models.items():
            # Preprocess new data using the same preprocessor
            preprocessor = self.pipeline_results['preprocessor']
            
            # Scale features using the saved scaler
            scaler_path = config.MODELS_DIR / f'scaler_{target_name}.pkl'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                new_data_scaled = scaler.transform(new_data)
                
                # Make predictions
                predictions[target_name] = model_info['model'].predict(new_data_scaled)
            else:
                predictions[target_name] = model_info['model'].predict(new_data)
        
        return predictions

def main():
    """Main function to run the complete pipeline"""
    
    print("🧪 Reservoir AI - Machine Learning Pipeline")
    print("🔬 Professional Grade Petrophysical Property Prediction")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = ReservoirAIPipeline(
        enable_feature_engineering=True,
        enable_shap=True
    )
    
    try:
        results = pipeline.run_complete_pipeline()
        
        # Save pipeline results
        joblib.dump(results, config.RESULTS_DIR / 'pipeline_results.pkl')
        print(f"\n💾 Pipeline results saved to: {config.RESULTS_DIR / 'pipeline_results.pkl'}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        return None

if __name__ == "__main__":
    # Run the complete pipeline
    results = main()
