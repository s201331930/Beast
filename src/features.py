import pandas as pd
import numpy as np
import ta
from textblob import TextBlob
import yfinance as yf

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe.
    """
    df = df.copy()
    
    # Ensure 'Close' is available
    if 'Close' not in df.columns:
        raise ValueError("Dataframe must contain 'Close' column")

    # Moving Averages
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    
    # Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['SMA_50']
    
    # Volume Indicators
    if 'Volume' in df.columns:
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std()
    
    return df

def fetch_market_context(start_date, end_date):
    """
    Fetches VIX and Oil prices as market context.
    """
    print("Fetching market context (VIX, Oil)...")
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date)['Close']
        oil = yf.download("CL=F", start=start_date, end=end_date)['Close']
        
        context_df = pd.DataFrame({'VIX': vix, 'Oil': oil})
        if isinstance(context_df.columns, pd.MultiIndex):
             context_df.columns = context_df.columns.get_level_values(0)
             
        return context_df
    except Exception as e:
        print(f"Error fetching market context: {e}")
        return pd.DataFrame()

def simulate_sentiment_data(dates):
    """
    Simulates sentiment data (e.g., from Tweets/News) for demonstration.
    In a real scenario, this would involve NLP on scraped text.
    """
    np.random.seed(42)
    sentiment_scores = np.random.normal(0, 0.5, size=len(dates))
    # Add some autocorrelation to make it look realistic
    for i in range(1, len(sentiment_scores)):
        sentiment_scores[i] = 0.8 * sentiment_scores[i-1] + 0.2 * np.random.normal(0, 0.5)
        
    return pd.Series(sentiment_scores, index=dates, name='Sentiment_Score')

def prepare_features(df):
    """
    Main function to prepare all features.
    """
    print("Generating features...")
    df = add_technical_indicators(df)
    
    # Get date range for context data
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # Fetch VIX and Oil (merge on Date)
    context_df = fetch_market_context(start_date, end_date)
    
    # Handle index for merging
    df.set_index('Date', inplace=True)
    
    # Merge context
    if not context_df.empty:
        df = df.join(context_df, how='left')
    
    # Simulate Sentiment
    df['Sentiment'] = simulate_sentiment_data(df.index)
    
    # Fill NAs (forward fill for context, 0 for others if appropriate or drop)
    df.ffill(inplace=True)
    df.dropna(inplace=True) # Drop initial rows with NaNs from rolling windows
    
    return df
