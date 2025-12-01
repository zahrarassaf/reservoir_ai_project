# SPE9 Technical Specifications

## Project Overview

The SPE9 Comparative Solution Project is a benchmark case for validating black-oil reservoir simulators. This implementation provides a complete, professional-grade simulation environment.

## Technical Specifications

### Grid Properties
- **Dimensions**: 24 × 25 × 15 cells
- **Total Cells**: 9,000
- **Cell Size**: 300 ft × 300 ft in x-y plane
- **Layer Thickness**: Variable (20-100 ft)
- **Porosity**: 0.08-0.17 fraction
- **Permeability**: Heterogeneous (0.01-10,000 mD)

### Fluid Properties
- **Oil**: Black oil with solution gas
- **Water**: Slightly compressible
- **Gas**: Real gas with compressibility
- **Initial Conditions**: Saturated oil at bubble point

### Well Configuration
- **Total Wells**: 26
- **Injectors**: 1 (water injection)
- **Producers**: 25 (oil production)
- **Completion**: Partial perforation in layers 2-4

### Simulation Parameters
- **Time Period**: 900 days
- **Timesteps**: Adaptive with maximum 10 days
- **Production Phases**:
  - Phase 1 (0-300 days): 1500 STB/day
  - Phase 2 (300-360 days): 100 STB/day
  - Phase 3 (360-900 days): 1500 STB/day
- **Injection**: 5000 STB/day water

### Physical Models
- **Rock**: Compressible with 4e-6 psi⁻¹
- **Fluid Flow**: Three-phase black oil
- **Relative Permeability**: Stone's Model II
- **Capillary Pressure**: Included
- **PVT**: Pressure-dependent properties

## Validation Criteria

### Expected Results (from SPE9 paper)
- **Final Oil Production**: ~7.5 MMSTB
- **Recovery Factor**: ~10%
- **Water Cut Development**: 0-50%
- **GOR Behavior**: Increasing with time

### Tolerance Levels
- **Production Rates**: ±5%
- **Cumulative Production**: ±2%
- **Pressure**: ±100 psi
- **Saturations**: ±0.05 fraction

## File Formats

### Input Files
- **.DATA**: Main simulation deck
- **.INC**: Include files (grid, properties)
- **.DATA**: External data files

### Output Files
- **.UNRST**: Restart files
- **.SMSPEC**: Summary data
- **.EGRID**: Grid geometry
- **.INIT**: Initial conditions

## References

1. Killough, J.E. (1995). "Ninth SPE Comparative Solution Project"
2. SPE Comparative Solution Project Documentation
3. OPM/Flow Simulator User Manual
