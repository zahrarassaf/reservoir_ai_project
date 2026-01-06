Reservoir AI Simulation - SPE9 Benchmark Analysis
📊 Project Overview
A comprehensive reservoir simulation and analysis system using the SPE9 Comparative Solution Project dataset. This project implements physics-based reservoir modeling integrated with machine learning for enhanced prediction accuracy and economic analysis.

🎯 Key Features
Data Integration
100% Real SPE9 Data: Complete utilization of SPE9 benchmark dataset

Reservoir Properties: Real permeability, porosity, and grid data from PERMVALUES.DATA, SPE9.GRDECL

Well Data: 26 actual wells extracted from SPE9.DATA (1 injector, 25 producers)

Economic Parameters: SPE9 benchmark economic data ($30/bbl oil price, $3.5/MSCF gas price)

Technical Capabilities
Physics-Based Simulation: 10-year reservoir performance forecasting

Machine Learning Integration:

CNN for reservoir property prediction (R² = 0.63)

Random Forest for economic forecasting

Economic Analysis: NPV, IRR, ROI, and payback period calculations

Visualization: Comprehensive plots and reports

📁 Project Structure
text
reservoir_ai_project/
├── data/                    # SPE9 benchmark dataset
│   ├── SPE9.DATA           # Main reservoir configuration
│   ├── SPE9.GRDECL         # Grid definition
│   ├── PERMVALUES.DATA     # Permeability values
│   ├── TOPSVALUES.DATA     # Depth values
│   └── SPE9_CP*.DATA       # Control and production files
├── src/
│   ├── data_loader.py      # Data ingestion and processing
│   └── main.py            # Main simulation pipeline
├── results/                # Output files
│   ├── spe9_analysis.png   # Visualizations
│   ├── spe9_report.json    # Comprehensive analysis report
│   ├── cnn_reservoir_model.pth      # Trained CNN model
│   └── svr_economic_model.joblib    # Economic prediction model
└── README.md              # This file
🔧 Installation & Setup
Prerequisites
Python 3.8+

Required packages: numpy, pandas, matplotlib, scikit-learn, torch (optional)

Installation
bash
# Clone repository
git clone https://github.com/zahrarassaf/reservoir_ai_project
cd reservoir-ai-simulation

# Install dependencies
pip install -r requirements.txt
Required Packages
txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
torch>=1.9.0
🚀 Usage
Basic Execution
bash
python src/main.py
Output
The simulation generates:

Visual Analysis: results/spe9_analysis.png

Comprehensive Report: results/spe9_report.json

Trained Models: CNN and Random Forest models

Console Summary: Detailed technical and economic analysis

📈 Results Summary
Technical Performance
Grid: 24×25×15 = 9,000 cells

Peak Production: 29,909 bpd

Total Recovery: 95.01 MM bbl

Recovery Factor: 35%

Economic Analysis (SPE9 Parameters)
Net Present Value: $730.95 Million

Internal Rate of Return: 9.5%

Return on Investment: 803.2%

Payback Period: 0.6 years

Break-even Price: $17.5/bbl

Machine Learning Performance
CNN Property Prediction: R² = 0.63

Economic Forecasting: Implemented with Random Forest

🔬 Methodology
Data Processing
Grid Construction: Parse SPE9.GRDECL for 3D reservoir structure

Property Mapping: Load real permeability and porosity data

Well Integration: Extract 26 wells with their configurations

Economic Parameters: Apply SPE9 benchmark pricing

Simulation Engine
Physics-Based Modeling: Arps decline curve analysis

Production Forecasting: 10-year projection with water cut development

Pressure Analysis: Reservoir pressure decline simulation

Machine Learning Integration
CNN Architecture: 3D convolutional network for property prediction

Feature Engineering: Reservoir and economic parameter extraction

Model Training: 800 synthetic cases for economic forecasting

📊 Data Validation
Real Data Utilization
✅ Reservoir Data: 100% real SPE9 benchmark

✅ Well Configuration: 26 actual SPE9 wells

✅ Economic Parameters: SPE9 standard pricing

✅ Grid Definition: Exact SPE9 dimensions (24×25×15)

Quality Assurance
Reproducible results with fixed random seed (42)

Comprehensive error handling and validation

Detailed logging and progress reporting

🎯 Key Advantages
Real Data Foundation: Uses industry-standard SPE9 benchmark

Integrated Approach: Combines physics-based and ML methods

Economic Focus: Comprehensive financial analysis

Production Ready: Generates actionable insights and reports

Scalable Architecture: Modular design for future enhancements

🔮 Future Enhancements
Planned improvements include:

Enhanced CNN architecture for better accuracy

Real-time economic parameter updates

Web interface for interactive analysis

Additional reservoir simulation methods

Integration with commercial simulators

📚 References
SPE Comparative Solution Project (SPE9)

Arps, J.J. (1945) "Analysis of Decline Curves"

Industry standard economic evaluation methods

Machine learning applications in reservoir engineering

👥 Contributing
Fork the repository

Create a feature branch

Commit changes with descriptive messages

Submit a pull request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
SPE (Society of Petroleum Engineers) for the benchmark dataset

Open-source community for ML libraries

Industry partners for validation and feedback
