# Reservoir AI & Digital Twin Project

A research-oriented AI-driven digital twin framework for reservoir simulation, uncertainty quantification, and production optimization using the SPE9 benchmark.

---

## Project Overview

This project presents an academic research framework that integrates physics-based reservoir simulation with machine learning and reinforcement learning to study decision-making under geological and economic uncertainty. Using the real SPE9 benchmark dataset, the work focuses on building a scalable and extensible digital twin suitable for further MSc/PhD-level research and publication.

The primary objective is not field deployment, but methodological development: combining physical models, data-driven surrogates, and optimization algorithms within a unified pipeline.

---

##  Research Contributions

* Development of a physics-informed digital twin for the SPE9 reservoir benchmark
* Integration of machine learning surrogate models to accelerate reservoir and economic predictions
* Uncertainty-aware economic evaluation using Monte Carlo simulation
* Reinforcement learning–based production strategy optimization under uncertainty
* Modular framework designed for reproducibility and academic extension

---

## 🧱 System Components

### 1. Physics-Based Reservoir Simulation

* SPE9 benchmark dataset (24 × 25 × 15 grid; 9000 cells)
* 26 production and injection wells
* Reservoir properties: permeability, porosity, pressure, saturation
* Full-field simulation workflow using Eclipse-format inputs

### 2. Machine Learning Models

* **CNN Surrogate Model**: 3D convolutional neural network for spatial reservoir property prediction
* **Random Forest Models**: Economic and performance metric prediction (NPV, IRR, ROI, payback period)
* Feature importance analysis to identify dominant geological and operational drivers

### 3. Digital Twin Framework

* Periodic state updating using simulated real-time data streams
* Data-quality monitoring and anomaly detection
* Predictive forecasting of reservoir and economic performance
* Interactive visual dashboards for analysis

### 4. Economic & Uncertainty Analysis

* Monte Carlo simulation (1000 scenarios)
* Sensitivity and scenario analysis (oil price, permeability, well configuration)
* Probabilistic distributions of key economic indicators

### 5. Reinforcement Learning Optimization

* Deep Q-Network (DQN) agent for production control
* State: reservoir pressure, production rates, recovery factor
* Actions: adjustment of injection and production rates
* Reward function combining NPV and recovery performance

---

## 📊 Representative Results (Benchmark Case)

These results demonstrate the analytical capability of the framework rather than guaranteed field performance:

| Metric                        | Value                              |
| ----------------------------- | ---------------------------------- |
| Net Present Value (NPV)       | up to ~$650M (benchmark scenarios) |
| Internal Rate of Return (IRR) | ~9–10%                             |
| Recovery Factor               | ~45%                               |
| Economic Success Probability  | ~94%                               |

Model performance was consistent with SPE9 benchmark expectations, with surrogate models achieving high predictive accuracy (R² > 0.94 for key economic indicators).

---

## 🏗️ Repository Structure

```
Reservoir AI Project/
├── src/
│   ├── ml/                     # Machine learning models
│   ├── physics/                # Physics-based simulation
│   └── utils/                  # Utilities
├── data/                       # SPE9 benchmark data
├── notebooks/                  # Jupyter analysis notebooks
├── results/                    # Generated outputs and reports
├── main.py                     # Main execution script
├── digital_twin.py             # Digital twin implementation
├── uncertainty_analysis.py     # Monte Carlo analysis
├── rl_reservoir_optimizer.py   # Reinforcement learning module
└── requirements.txt
```

---

## Usage (Research-Oriented)

```bash
python main.py                     # Run baseline simulation
python digital_twin.py             # Execute digital twin workflow
python uncertainty_analysis.py     # Perform uncertainty analysis
python rl_reservoir_optimizer.py   # Run RL optimization
```

---

## 🧪 Model Performance Summary

| Model         | Target       | R²   |
| ------------- | ------------ | ---- |
| Random Forest | NPV          | 0.94 |
| Random Forest | IRR          | 0.97 |
| Random Forest | ROI          | 0.99 |
| CNN           | Permeability | 0.94 |

---

##  Research Context

This project was developed as part of an academic research initiative at the intersection of reservoir engineering and artificial intelligence. The framework is intentionally modular to support extension toward peer-reviewed publication, including:

* Advanced surrogate modeling
* Multi-objective reinforcement learning
* Field-scale digital twin generalization

---

##  Novelty Statement

The novelty of this work lies in the unified integration of physics-based reservoir simulation, machine learning surrogates, uncertainty quantification, and reinforcement learning within a single digital twin framework using the SPE9 benchmark. Unlike many existing studies that focus on isolated components (e.g., surrogate modeling or economic optimization), this project emphasizes end-to-end decision-making under uncertainty, with explicit coupling between geological properties, production strategy, and economic outcomes.

---

## 📚 Related Work

This work is informed by and related to prior research in the following areas:

* SPE9 Benchmark studies for reservoir simulation and optimization (Society of Petroleum Engineers)
* Physics-informed machine learning for subsurface modeling
* Surrogate modeling for accelerated reservoir simulation
* Reinforcement learning applications in reservoir management and production control

A curated list of references and DOIs will be included as part of the manuscript preparation for peer-reviewed publication.

---

##  Experimental Setup

* Reservoir model: SPE9 benchmark (24 × 25 × 15 grid, 9000 cells)
* Wells: 26 producers and injectors
* Simulation horizon: multi-year production forecast
* Machine learning training: train/validation/test split with cross-validation where applicable
* Monte Carlo analysis: 1000 realizations across 10 uncertain geological and economic parameters
* Evaluation metrics: R², MAE for surrogate models; NPV, IRR, recovery factor for optimization performance

---

## Limitations

* Benchmark-specific results limited to the SPE9 dataset
* Simplified economic assumptions compared to full field-development studies
* Reinforcement learning trained on simulated environments rather than live field data

Future work will address these limitations by extending the framework to multiple reservoirs, incorporating more complex economic models, and validating RL strategies on field-scale or synthetic real-time data.

---

##  Future Research Directions

* Extension to multi-reservoir and field-scale optimization
* Incorporation of transformer-based surrogate models
* Coupling with live or synthetic streaming production data
* Development of uncertainty-aware multi-agent RL strategies

---

##  License

MIT License

---



*This repository is intended for academic and research use. Results are benchmark-specific and should not be interpreted as guaranteed field performance.*
