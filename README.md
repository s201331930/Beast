# Stock Market Anomaly Detection - RKLB Case Study

This project implements a comprehensive anomaly detection system to predict big stock moves, using RKLB (Rocket Lab) as a case study.

## Objectives
- Flag potential big moves (rallies) early.
- Utilize price, volume, and alternative data (simulated/public).
- Employ statistical, ML, and Deep Learning models.

## Models Implemented
- Statistical: Z-Score, Moving Average, Bollinger Bands
- Machine Learning: Isolation Forest, One-Class SVM
- Deep Learning: LSTM Autoencoders
- Cyclical/Mean Reversion: RSI, Hurst Exponent

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run data loader: `python src/data_loader.py`
3. Run main analysis: `python src/main.py`
