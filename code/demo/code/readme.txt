# NYC Taxi Demand Forecasting & Dispatch Demo

## Overview

This repository contains two Python scripts that together implement a spatiotemporal taxi-demand forecasting pipeline and an interactive dispatch GUI:

1. **Demo script** (`demo.py`):  
   - Loads historical taxi‐trip data  
   - Trains a Multi-Scale RNN (GRU+LSTM+Transformer) on a rolling 30-day window  
   - Refines the RNN outputs with a GraphSAGE GNN over the zone OD-flow graph  
   - Saves per-zone predictions and generates an interactive Folium heatmap (`nyc_taxi_prediction_map.html`)

2. **Dispatch GUI** (`window_map.py`):  
   - Embeds the generated heatmap in a PyQt5 window  
   - Allows the operator to enter a Zone ID and receive real-time dispatch recommendations based on predicted supply–demand gaps

---

## Requirements

- Python 3.8+  
- PyTorch  
- PyTorch Geometric  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib  
- PyArrow  
- Folium  
- Branca  
- PyQt5  
- PyQtWebEngine  

You can install all dependencies with:

```bash
pip install torch torch_geometric pandas numpy scikit-learn matplotlib pyarrow folium branca PyQt5 PyQtWebEngine
