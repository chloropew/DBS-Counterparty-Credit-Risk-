# Sea Surface Temperature (SST) Forecasting with Quantum Reservoir Computing (QRC)

This project benchmarks the performance of a simple forecasting model using **Quantum Reservoir Computing (QRC)** embeddings compared to a classical linear neural network, focusing on **sea surface temperature (SST)** prediction.

## Overview

Due to memory constraints, using data from 100 regions caused the system to crash. Even downscaling to one region resulted in instability. To establish a minimal and reliable benchmark, we designed a simpler setup using only **1 input feature (SST)** and **1 output region**.

## Experiment Setup

- **Input Feature**: Sea Surface Temperature (SST) only  
- **Prediction Target**: SST at a single selected region  
- **Training Duration**: 10 months  
- **Input Window**: 14 days  
- **Forecast Horizon**: 7 days  

## Model Comparison

We implemented two models:
1. **Linear Neural Network** (baseline)
2. **Linear Neural Network with QRC Embedding**

Refer to `Comparison.png` for the plot comparing the predictions.

- The **QRC-enhanced model** (green) demonstrates higher prediction accuracy than the classical model.
- The **actual SST values** are shown in blue for reference.

## Key Insights

- This result establishes a benchmark showing that **QRC can outperform classical models** in SST forecasting accuracy.
- Leveraging the principles of **quantum superposition**, the QRC model is expected to be more **computationally efficient** by enabling **parallel information processing**.

---

This benchmark provides an early indication that QRC holds promise for time-series forecasting tasks, particularly in environmental and geophysical domains like SST prediction.
