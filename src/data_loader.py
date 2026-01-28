import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_stock_data(ticker="RKLB", start_date="2020-01-01", end_date=None):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Ensure MultiIndex columns are handled if present (yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.reset_index(inplace=True)
        print(f"Successfully fetched {len(data)} rows.")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_data(data, filepath="data/raw_data.csv"):
    """
    Saves dataframe to CSV.
    """
    if data is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

if __name__ == "__main__":
    df = fetch_stock_data()
    save_data(df)
