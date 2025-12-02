def run_simulation_pipeline(data: Dict[str, Any], 
                          configs: Dict[str, Any],
                          modules: Dict[str, Any]) -> bool:
    """Run the complete simulation pipeline."""
    
    # Initialize plots_generated at the beginning
    plots_generated = 0
    
    try:
        # ... [بقیه کد] ...
        
        # Step 4: Generate plots
        if modules['PlotGenerator']:
            try:
                logger.info("Step 4: Generating plots...")
                
                # Initialize plot generator correctly
                plot_generator = modules['PlotGenerator'](serializable_results)
                
                # Try different plot methods
                plot_methods = [
                    ('plot_pressure_distribution', 'pressure_plot'),
                    ('plot_production_history', 'production_plot'),
                    ('plot_saturation', 'saturation_plot'),
                    ('plot_well_performance', 'well_plot')
                ]
                
                for method_name, plot_name in plot_methods:
                    if hasattr(plot_generator, method_name):
                        try:
                            fig = getattr(plot_generator, method_name)()
                            if fig:
                                plot_path = results_dir / f"{plot_name}_{timestamp}.png"
                                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                                plots_generated += 1
                                logger.info(f"Generated: {plot_path}")
                        except Exception as e:
                            logger.debug(f"Could not generate {method_name}: {e}")
                
                logger.info(f"✅ Generated {plots_generated} plots")
                
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
                # plots_generated remains 0
        
        # ... [بقیه کد] ...
        
        logger.info(f"📈 Plots generated: {plots_generated}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simulation pipeline failed: {e}", exc_info=True)
        return False
