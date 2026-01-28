import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_market_data(ticker="RKLB", start_date="2020-01-01", end_date=None):
    """
    Fetches market data for the target ticker and related macro indicators.
    """
    print(f"Fetching data for {ticker}...")
    
    # 1. Target Stock
    df_stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df_stock.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Flatten multi-level columns if present (common in new yfinance)
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = df_stock.columns.get_level_values(0)
    
    df_stock = df_stock.rename(columns={
        "Open": "open", "High": "high", "Low": "low", 
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })
    
    # 2. Macro Indicators
    # VIX: Volatility Index
    # CL=F: Crude Oil Futures
    # ^GSPC: S&P 500
    macro_tickers = {
        "^VIX": "vix",
        "CL=F": "oil",
        "^GSPC": "sp500"
    }
    
    macro_data = {}
    for symbol, name in macro_tickers.items():
        print(f"Fetching {name} ({symbol})...")
        try:
            df_macro = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df_macro.columns, pd.MultiIndex):
                df_macro.columns = df_macro.columns.get_level_values(0)
            
            # We mostly care about the Close price for macro indicators
            macro_data[name] = df_macro["Close"]
        except Exception as e:
            print(f"Warning: Could not fetch {symbol}: {e}")
            
    # Join macro data
    for name, series in macro_data.items():
        df_stock[name] = series
        
    # Forward fill macro data to handle trading holidays or mismatched timestamps
    df_stock = df_stock.ffill()
    
    return df_stock

def generate_mock_alternative_data(df):
    """
    Generates mock alternative data (Sentiment, Search Volume) for demonstration.
    In a real scenario, this would be an API call to Twitter/X or Google Trends.
    """
    np.random.seed(42)
    n = len(df)
    
    # Simulate "Tweet Volume" correlated with volatility and volume
    # Random walk + shocks based on volume spikes
    tweet_vol = np.random.normal(1000, 100, n)
    # Add spikes where volume is high
    vol_z = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    tweet_vol += np.maximum(0, vol_z) * 500
    df['twitter_volume'] = tweet_vol.astype(int)
    
    # Simulate "Sentiment" (-1 to 1)
    # Mean reverting process
    sentiment = np.zeros(n)
    for i in range(1, n):
        sentiment[i] = 0.9 * sentiment[i-1] + np.random.normal(0, 0.1)
    
    # Add some correlation to future returns (the "alpha") - artificially making it predictive for the demo
    # Future return (3 days)
    future_ret = df['close'].shift(-3) / df['close'] - 1
    # Add signal to sentiment (leaking future info slightly to ensure model picks something up, 
    # but strictly this is just for "mocking" a good signal source)
    # In reality, we wouldn't look ahead.
    # Let's just make it correlated with current price momentum for realism without lookahead bias
    momentum = df['close'].pct_change(5).fillna(0)
    sentiment += momentum * 2
    df['sentiment_score'] = np.clip(sentiment, -1, 1)
    
    # Google Trends (0-100)
    # Correlated with high volatility
    trends = np.random.normal(20, 5, n)
    trends += np.abs(df['close'].pct_change()) * 500
    df['google_trends'] = np.clip(trends, 0, 100)
    
    return df

def load_and_process_data(ticker="RKLB"):
    cache_file = f"data/{ticker}_data.csv"
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        df = fetch_market_data(ticker)
        df = generate_mock_alternative_data(df)
        df.to_csv(cache_file)
        print(f"Data saved to {cache_file}")
        
    return df

if __name__ == "__main__":
    df = load_and_process_data()
    print(df.head())
    print(df.describe())
