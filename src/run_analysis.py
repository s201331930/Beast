import pandas as pd
import numpy as np
import sys
import os

# Ensure src module is importable
sys.path.append(os.getcwd())

from src.data_loader import load_and_process_data
from src.feature_engineering import prepare_features
from src.models import backtest_models

def run_analysis(ticker="BABA", horizon=5):
    print(f"\n{'='*50}")
    print(f"Running Anomaly Prediction Analysis for {ticker}")
    print(f"{'='*50}\n")
    
    # 1. Load Data
    # We force a fresh download or separate cache for the new ticker
    cache_file = f"data/{ticker}_data.csv"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    df = load_and_process_data(ticker)
    
    # 2. Prepare Features
    print("Generating Features...")
    df = prepare_features(df)
    
    # 3. Run Backtest
    print("Running Backtest and Signal Generation...")
    results = backtest_models(df, horizon=horizon)
    
    # 4. Extract Signals
    signals = results[results['final_signal'] == 1].copy()
    
    print(f"\n--- Detected High-Confidence Rally Signals for {ticker} ---")
    if len(signals) > 0:
        # Show Date, Close Price, Probability, and actual return (if known)
        display_cols = ['close', 'prob_rally', 'actual_return_horizon']
        signals['return_pct'] = signals['actual_return_horizon'] * 100
        
        print(signals[display_cols].to_string())
        
        print(f"\nTotal Signals Found: {len(signals)}")
        
        # Calculate win rate for this specific ticker
        win_rate = (signals['actual_return_horizon'] > 0).mean()
        avg_return = signals['actual_return_horizon'].mean()
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Avg Return per Trade ({horizon} days): {avg_return*100:.2f}%")
        
    else:
        print("No high-confidence signals were flagged for this ticker in the test period.")
        
    return results

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BABA"
    run_analysis(ticker)
