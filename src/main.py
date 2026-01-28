import pandas as pd
import numpy as np
import warnings
from src.data_loader import fetch_stock_data, save_data
from src.features import prepare_features
from src.models.anomaly_detectors import StatisticalDetector, IsolationForestDetector, OneClassSVMDetector, LSTMAnomalyDetector
from src.backtest import Backtester
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def main():
    print("Starting Stock Market Anomaly Detection Pipeline for RKLB...")
    
    # 1. Data Acquisition
    # Fetching longer history if available, or just what we can get
    df = fetch_stock_data(ticker="RKLB", start_date="2020-01-01")
    if df is None:
        return
    
    # 2. Feature Engineering
    df = prepare_features(df)
    save_data(df, "data/processed_data.csv")
    
    # Define feature columns for ML models
    feature_cols = ['Log_Return', 'Volatility_20', 'RSI', 'MACD', 'BB_Width', 'Sentiment']
    if 'VIX' in df.columns:
        feature_cols.append('VIX')
    if 'Oil' in df.columns:
        feature_cols.append('Oil')
    if 'Volume_Ratio' in df.columns:
        feature_cols.append('Volume_Ratio')
        
    print(f"Using features: {feature_cols}")
    
    results = {}
    
    # 3. Modeling & 4. Backtesting
    
    # --- Statistical ---
    print("\nRunning Statistical Detector...")
    stat_detector = StatisticalDetector(window=20, threshold=2.5)
    df['Stat_Signal'] = stat_detector.detect(df)
    
    bt_stat = Backtester(df, 'Stat_Signal')
    results['Statistical'] = bt_stat.evaluate()
    bt_stat.simulate_trading()
    bt_stat.plot_results()
    
    # --- Isolation Forest ---
    print("\nRunning Isolation Forest...")
    iso_detector = IsolationForestDetector(contamination=0.05)
    df['IsoForest_Signal'] = iso_detector.fit_predict(df, feature_cols)
    
    bt_iso = Backtester(df, 'IsoForest_Signal')
    results['IsolationForest'] = bt_iso.evaluate()
    bt_iso.simulate_trading()
    bt_iso.plot_results()
    
    # --- One-Class SVM ---
    print("\nRunning One-Class SVM...")
    svm_detector = OneClassSVMDetector(nu=0.05)
    df['SVM_Signal'] = svm_detector.fit_predict(df, feature_cols)
    
    bt_svm = Backtester(df, 'SVM_Signal')
    results['OneClassSVM'] = bt_svm.evaluate()
    bt_svm.simulate_trading()
    bt_svm.plot_results()
    
    # --- LSTM Autoencoder ---
    print("\nRunning LSTM Autoencoder (Deep Learning)...")
    lstm_detector = LSTMAnomalyDetector(epochs=20) # Low epochs for demo speed
    # LSTM returns array with padded zeros, ensure it matches
    lstm_signals = lstm_detector.fit_predict(df, feature_cols)
    # Align lengths just in case
    if len(lstm_signals) == len(df):
        df['LSTM_Signal'] = lstm_signals
        
        bt_lstm = Backtester(df, 'LSTM_Signal')
        results['LSTM'] = bt_lstm.evaluate()
        bt_lstm.simulate_trading()
        bt_lstm.plot_results()
    else:
        print(f"Shape mismatch: LSTM signals {len(lstm_signals)} vs DF {len(df)}")
    
    # Summary
    print("\n--- Summary of Model Performance (Precision in predicting Big Moves) ---")
    summary_df = pd.DataFrame(results).T
    print(summary_df)
    
    save_data(df, "data/final_results.csv")
    print("Analysis Complete. Results saved to data/final_results.csv")

if __name__ == "__main__":
    main()
