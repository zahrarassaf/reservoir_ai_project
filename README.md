AI-Driven Subsurface Flow Modeling for Sustainable Resource & Water Management


This project develops machine-learning models (CNN-LSTM and SVR) to predict reservoir production behavior—pressure, water cut, and flow rates—using synthetic and SPE9 benchmark data.
While originally designed for petroleum reservoir studies, the modeling approach has direct applications in environmental subsurface analysis, produced-water management, and CO₂ storage monitoring.

⭐ 1. Project Overview

This project builds an end-to-end ML pipeline to forecast dynamic subsurface behavior. The goal is to show how advanced temporal–spatial deep learning models can support more sustainable subsurface operations by improving prediction accuracy and reducing unnecessary extraction activities.

The workflow includes:

Data preprocessing

Feature engineering

CNN feature extraction

LSTM temporal modeling

SVR regression

Model comparison & evaluation

⭐ 2. Environmental Relevance (Why this Matters)

Although developed on reservoir data, the methodology directly supports environmental and sustainability applications, including:

✔ Produced-Water Management

Accurate water-cut prediction helps reduce water disposal volumes and environmental contamination risk.

✔ Energy Efficiency & Reduced Extraction Footprint

Better production forecasting prevents unnecessary field operations, reducing energy use and operational emissions.

✔ CO₂ Sequestration Monitoring

CNN-LSTM architectures used here are directly applicable to CO₂ plume tracking, pressure monitoring, and leakage-risk assessment in carbon storage sites.

✔ Groundwater & Subsurface Hydrology Modeling

The same modeling pipeline can simulate groundwater flow or contaminant transport with appropriate datasets.

⭐ 3. Dataset
Synthetic Dataset (included)

10,000+ simulation samples

Features: pressure, water cut, porosity, permeability…

Time-series format for sequential modeling

SPE9 Dataset (OPM) – optional external validation

Industry-standard benchmark

Used to compare model performance on realistic subsurface scenarios

⭐ 4. Methods & Models
🔹 CNN-LSTM

CNN extracts spatial reservoir features

LSTM captures temporal dependencies

Suitable for complex physical systems with time-varying patterns

🔹 Support Vector Regression (SVR)

Models nonlinear relationships

Performs strongly on structured engineering data

🔹 Baseline Models

Linear Regression

Random Forest

XGBoost

⭐ 5. Results

CNN-LSTM RMSE: X.XX

SVR RMSE: X.XX

Improvement vs baseline: XX%

Best performance achieved by: CNN-LSTM (temporal + spatial capability)

⭐ 6. Applications

This modeling pipeline can be applied to:

Energy systems forecasting

Water sustainability modeling

CO₂ injection & storage monitoring

Groundwater hydrology

Environmental risk assessment

Subsurface contamination simulations

⭐ 7. How to Run the Project
git clone https://github.com/Zahrarasaf/reservoir_ai_project
cd reservoir_ai_project
pip install -r requirements.txt
python train_models.py

⭐ 8. Project Structure
/data            → Synthetic datasets  
/models          → CNN-LSTM, SVR, baselines  
/notebooks       → EDA & model experimentation  
/scripts         → Preprocessing & training  
README.md        → Project documentation  

⭐ 9. Skills Demonstrated

Time-series ML

Deep learning: CNN + LSTM hybrid models

Regression modelling

Feature engineering

Model tuning & evaluation

Scientific data processing

Environmental data science applications

## Dataset
- Synthetic dataset included for immediate execution
- SPE9 dataset (OPM): [GitHub link](https://github.com/OPM/opm-data/tree/master/spe9)

## Installation
```bash
pip install -r requirements.txt
