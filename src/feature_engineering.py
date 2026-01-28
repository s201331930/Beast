import pandas as pd
import numpy as np
import ta
from scipy.fft import fft, fftfreq

def add_technical_indicators(df):
    """
    Adds standard technical indicators.
    """
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
    df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # ATR (Average True Range) - Volatility
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Volume SMA
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df

def add_cyclical_features(df, lookback=60, top_k=3):
    """
    Adds features based on Fourier Transform to detect dominant cycles.
    """
    # We apply FFT on a rolling window of recent prices (detrended)
    
    # Initialize columns
    for k in range(top_k):
        df[f'cycle_period_{k}'] = np.nan
        df[f'cycle_mag_{k}'] = np.nan
        
    prices = df['close'].values
    
    # We need a sufficient window
    for i in range(lookback, len(df)):
        window = prices[i-lookback:i]
        
        # Detrend linear trend
        x = np.arange(len(window))
        p = np.polyfit(x, window, 1)
        trend = np.polyval(p, x)
        detrended = window - trend
        
        # FFT
        n = len(window)
        yf_fft = fft(detrended)
        xf = fftfreq(n, 1) # 1 day sampling
        
        # Get magnitudes for positive frequencies
        magnitudes = np.abs(yf_fft[:n//2])
        freqs = xf[:n//2]
        
        # Sort by magnitude (skip DC component at index 0)
        sorted_indices = np.argsort(magnitudes[1:])[::-1] + 1
        
        # Store top K dominant periods
        for k in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[k]
            freq = freqs[idx]
            if freq > 0:
                period = 1 / freq
                df.at[df.index[i], f'cycle_period_{k}'] = period
                df.at[df.index[i], f'cycle_mag_{k}'] = magnitudes[idx]
            
    return df

def add_statistical_features(df):
    """
    Adds Z-scores and other statistical anomalies.
    """
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling Z-score of returns (30 day window)
    roll_mean = df['returns'].rolling(window=30).mean()
    roll_std = df['returns'].rolling(window=30).std()
    df['z_score_ret'] = (df['returns'] - roll_mean) / roll_std
    
    # Hurst Exponent (simplified rolling proxy)
    # A true Hurst calc is slow, so we use a volatility ratio proxy
    # V_ratio = Variance(n*T) / (n * Variance(T)) - roughly checks mean reversion vs trend
    # Here we just use a simplified volatility ratio
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    df['vol_ratio'] = df['volatility_10'] / df['volatility_30']
    
    return df

def create_target(df, horizon=5, threshold=0.10):
    """
    Creates the target variable: 1 if max price in next 'horizon' days > current price * (1+threshold)
    """
    # Forward looking max price
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    future_max = df['high'].rolling(window=indexer).max()
    
    df['future_max_return'] = (future_max - df['close']) / df['close']
    df['target_rally'] = (df['future_max_return'] > threshold).astype(int)
    
    return df

def prepare_features(df):
    df = add_technical_indicators(df)
    df = add_statistical_features(df)
    df = add_cyclical_features(df)
    df = create_target(df)
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    from data_loader import load_and_process_data
    df = load_and_process_data()
    df = prepare_features(df)
    print(df[['close', 'rsi', 'cycle_period_0', 'target_rally']].tail(10))
