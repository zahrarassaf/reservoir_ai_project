AI-Driven Subsurface Flow Modeling for Sustainable Resource & Water Management

This project develops advanced machine-learning models (CNN-LSTM and SVR) to predict subsurface flow behavior—pressure evolution, water cut, and production rates—using both synthetic reservoir simulations and the SPE9 benchmark dataset.
Although originally inspired by reservoir engineering workflows, the modeling framework is designed with environmental subsurface monitoring, water-resource sustainability, and CO₂ storage operations in mind.

⭐ 1. Project Overview

This repository contains an end-to-end machine learning pipeline for forecasting dynamic subsurface behavior.
The objective is to demonstrate how temporal–spatial deep learning architectures can support more sustainable decision-making by improving prediction accuracy and reducing unnecessary surface operations.

Workflow Components

Data preprocessing & cleaning

Geoscience-inspired feature engineering

CNN-based spatial feature extraction

LSTM temporal prediction

SVR nonlinear regression

Full baseline comparison (Linear Regression, RF, XGBoost)

⭐ 2. Environmental Relevance (Why This Matters)

Although based on subsurface flow simulations, this methodology directly supports environmental data science, enabling:

✔ Produced-Water Management

More accurate water-cut prediction → better disposal planning → reduced environmental contamination risk.

✔ Energy Efficiency & Reduced Surface Footprint

Accurate forecasts reduce unnecessary extraction cycles, lowering energy consumption and operational emissions.

✔ CO₂ Sequestration Monitoring

CNN-LSTM architectures are highly suitable for:

Pressure-front tracking

CO₂ plume migration prediction

Leakage-risk early-warning systems

✔ Groundwater & Subsurface Hydrology

The same ML pipeline can be repurposed for:

Groundwater flow forecasting

Contaminant transport modeling

Water-resource sustainability planning

⭐ 3. Datasets
Synthetic Dataset (included)

10,000+ simulation samples

Features: pressure, water cut, porosity, permeability, saturation, etc.

Structured as multivariate time-series + spatial grids

SPE9 Dataset (OPM)

Industry-standard benchmark for subsurface simulations
Used to validate model robustness under realistic geological conditions.

⭐ 4. Methods & Models
🔹 CNN-LSTM Hybrid Model

CNN extracts spatial reservoir patterns

LSTM models temporal dependencies

Highly effective for sequence-to-sequence forecasting in physical systems

🔹 Support Vector Regression (SVR)

Captures nonlinear relationships in structured engineering datasets

🔹 Baselines

Linear Regression

Random Forest Regressor

XGBoost Regressor

⭐ 5. Results

(Replace X.XX with your actual metrics once model is run.)

CNN-LSTM RMSE: X.XX

SVR RMSE: X.XX

Improvement over baseline: XX%

Best model: CNN-LSTM

⭐ 6. Applications

This modeling framework can be applied to:

Sustainable resource management

Groundwater hydrology modeling

CO₂ injection & long-term storage monitoring

Environmental risk assessment

Produced-water prediction & treatment planning

Energy system forecasting

⭐ 7. How to Run the Project
git clone https://github.com/Zahrarasaf/reservoir_ai_project
cd reservoir_ai_project
pip install -r requirements.txt
python train_models.py

⭐ 8. Project Structure
/data            → Synthetic datasets
/models          → CNN-LSTM, SVR, and baselines
/notebooks       → EDA, experiments, visualizations
/scripts         → Preprocessing & training scripts
README.md        → Project documentation

⭐ 9. Skills Demonstrated

Time-series machine learning

Deep learning: CNN + LSTM integration

Nonlinear regression (SVR, XGBoost, RF)

Scientific data engineering

Environmental data science applications

Model evaluation & tuning

Subsurface simulation analysis
