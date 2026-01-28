"""
Comprehensive Data Collection Module for Stock Anomaly Detection
Collects price, volume, options, VIX, oil, and alternative data sources.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import requests
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class MarketDataCollector:
    """
    Collects market data from various sources including:
    - Price and volume data
    - Options data (put/call ratios)
    - VIX (volatility index)
    - Oil prices (energy sector correlation)
    - Related ETFs and indices
    """
    
    def __init__(self, ticker: str, start_date: str, end_date: Optional[str] = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = {}
        
    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        logger.info(f"Fetching price data for {self.ticker}")
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=self.start_date, end=self.end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Calculate additional price metrics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # True Range and ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        self.data['price'] = df
        logger.info(f"Fetched {len(df)} rows of price data")
        return df
    
    def fetch_vix(self) -> pd.DataFrame:
        """Fetch VIX (CBOE Volatility Index) data."""
        logger.info("Fetching VIX data")
        
        vix = yf.Ticker("^VIX")
        df = vix.history(start=self.start_date, end=self.end_date)
        df.columns = [f'vix_{col.lower()}' for col in df.columns]
        df['vix_change'] = df['vix_close'].pct_change()
        df['vix_ma20'] = df['vix_close'].rolling(window=20).mean()
        df['vix_zscore'] = (df['vix_close'] - df['vix_ma20']) / df['vix_close'].rolling(20).std()
        
        self.data['vix'] = df[['vix_close', 'vix_change', 'vix_ma20', 'vix_zscore']]
        return df
    
    def fetch_oil(self) -> pd.DataFrame:
        """Fetch crude oil futures data (USO ETF as proxy)."""
        logger.info("Fetching oil data")
        
        oil = yf.Ticker("USO")
        df = oil.history(start=self.start_date, end=self.end_date)
        df.columns = [f'oil_{col.lower()}' for col in df.columns]
        df['oil_change'] = df['oil_close'].pct_change()
        df['oil_ma20'] = df['oil_close'].rolling(window=20).mean()
        
        self.data['oil'] = df[['oil_close', 'oil_change', 'oil_ma20']]
        return df
    
    def fetch_related_assets(self) -> pd.DataFrame:
        """Fetch related ETFs and indices for correlation analysis."""
        logger.info("Fetching related assets")
        
        # Space/Aerospace related
        related_tickers = {
            'SPY': 'sp500',      # S&P 500
            'QQQ': 'nasdaq',     # NASDAQ 100
            'IWM': 'russell',    # Russell 2000 (small caps)
            'ARKX': 'space_etf', # ARK Space Exploration
            'XLI': 'industrials', # Industrial sector
        }
        
        related_data = {}
        for ticker, name in related_tickers.items():
            try:
                data = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)
                related_data[f'{name}_close'] = data['Close']
                related_data[f'{name}_returns'] = data['Close'].pct_change()
            except Exception as e:
                logger.warning(f"Could not fetch {ticker}: {e}")
        
        df = pd.DataFrame(related_data)
        self.data['related'] = df
        return df
    
    def fetch_options_data(self) -> pd.DataFrame:
        """
        Fetch options data to calculate put/call ratios and unusual activity.
        Note: Historical options data requires premium APIs. Using available data.
        """
        logger.info("Fetching options data")
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get available expiration dates
            expirations = stock.options
            
            if not expirations:
                logger.warning("No options data available")
                return pd.DataFrame()
            
            # Get current options chain for analysis
            options_data = []
            for exp in expirations[:5]:  # First 5 expirations
                try:
                    opt = stock.option_chain(exp)
                    calls = opt.calls
                    puts = opt.puts
                    
                    call_volume = calls['volume'].sum() if 'volume' in calls else 0
                    put_volume = puts['volume'].sum() if 'volume' in puts else 0
                    call_oi = calls['openInterest'].sum() if 'openInterest' in calls else 0
                    put_oi = puts['openInterest'].sum() if 'openInterest' in puts else 0
                    
                    options_data.append({
                        'expiration': exp,
                        'call_volume': call_volume,
                        'put_volume': put_volume,
                        'call_oi': call_oi,
                        'put_oi': put_oi,
                        'pcr_volume': put_volume / call_volume if call_volume > 0 else np.nan,
                        'pcr_oi': put_oi / call_oi if call_oi > 0 else np.nan
                    })
                except Exception as e:
                    logger.warning(f"Could not fetch options for {exp}: {e}")
            
            df = pd.DataFrame(options_data)
            self.data['options'] = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return pd.DataFrame()
    
    def calculate_synthetic_pcr(self) -> pd.DataFrame:
        """
        Calculate synthetic put/call ratio based on price movements and volatility.
        This is a proxy when real-time PCR data is not available.
        """
        if 'price' not in self.data:
            self.fetch_price_data()
        
        df = self.data['price'].copy()
        
        # Synthetic PCR based on bearish vs bullish signals
        # Higher values indicate more bearish sentiment
        df['synthetic_pcr'] = (
            (df['returns'] < 0).rolling(20).mean() +  # % negative days
            (df['close'] < df['close'].rolling(50).mean()).astype(int) * 0.5 +  # Below 50d MA
            df['volatility_20d'].rank(pct=True) * 0.3  # High volatility = fear
        )
        
        return df[['synthetic_pcr']]
    
    def collect_all_data(self) -> pd.DataFrame:
        """Collect all data sources and merge into single DataFrame."""
        logger.info("Collecting all market data")
        
        # Fetch all data
        self.fetch_price_data()
        self.fetch_vix()
        self.fetch_oil()
        self.fetch_related_assets()
        self.fetch_options_data()
        
        # Merge all data on date index
        merged = self.data['price'].copy()
        
        for key in ['vix', 'oil', 'related']:
            if key in self.data and not self.data[key].empty:
                merged = merged.join(self.data[key], how='left')
        
        # Add synthetic PCR
        pcr = self.calculate_synthetic_pcr()
        merged = merged.join(pcr, how='left')
        
        # Forward fill missing values
        merged = merged.ffill()
        
        self.merged_data = merged
        logger.info(f"Final dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
        
        return merged
    
    def save_data(self, filepath: str):
        """Save collected data to CSV."""
        if hasattr(self, 'merged_data'):
            self.merged_data.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
        else:
            logger.warning("No merged data available. Run collect_all_data() first.")


class SentimentDataCollector:
    """
    Collects sentiment data from various sources:
    - Twitter/X mentions and sentiment
    - Google Trends search interest
    - News sentiment
    """
    
    def __init__(self, ticker: str, company_name: str = "Rocket Lab"):
        self.ticker = ticker
        self.company_name = company_name
        self.keywords = [ticker, company_name, f"${ticker}"]
        
    def fetch_google_trends(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Google Trends data for the ticker and company name.
        Note: Requires pytrends which may have rate limits.
        """
        logger.info("Fetching Google Trends data")
        
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Build payload
            kw_list = [self.ticker, self.company_name]
            pytrends.build_payload(kw_list, timeframe=f'{start_date} {end_date}')
            
            # Get interest over time
            df = pytrends.interest_over_time()
            
            if not df.empty:
                df = df.drop('isPartial', axis=1, errors='ignore')
                df.columns = [f'gtrends_{col.lower().replace(" ", "_")}' for col in df.columns]
                
                # Calculate z-scores for anomaly detection
                for col in df.columns:
                    df[f'{col}_zscore'] = (df[col] - df[col].rolling(30).mean()) / df[col].rolling(30).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_sentiment(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic sentiment data based on price action.
        This models how retail sentiment typically follows price momentum.
        """
        logger.info("Generating synthetic sentiment data")
        
        df = pd.DataFrame(index=price_df.index)
        
        # Synthetic Twitter mentions (higher during volatile periods)
        np.random.seed(42)
        base_mentions = 100 + np.random.poisson(50, len(price_df))
        volatility_factor = price_df['volatility_20d'].fillna(0.3) * 500
        momentum_factor = abs(price_df['returns'].fillna(0)) * 1000
        
        df['twitter_mentions'] = base_mentions + volatility_factor.values + momentum_factor.values
        df['twitter_mentions_ma7'] = df['twitter_mentions'].rolling(7).mean()
        df['twitter_mentions_zscore'] = (
            (df['twitter_mentions'] - df['twitter_mentions'].rolling(30).mean()) / 
            df['twitter_mentions'].rolling(30).std()
        )
        
        # Synthetic sentiment score (-1 to 1, follows recent returns with lag)
        returns_5d = price_df['returns'].rolling(5).mean().fillna(0)
        noise = np.random.normal(0, 0.1, len(price_df))
        df['sentiment_score'] = np.clip(returns_5d.values * 10 + noise, -1, 1)
        df['sentiment_ma7'] = df['sentiment_score'].rolling(7).mean()
        
        # Synthetic news sentiment
        df['news_sentiment'] = df['sentiment_score'] * 0.8 + np.random.normal(0, 0.15, len(df))
        df['news_sentiment'] = df['news_sentiment'].clip(-1, 1)
        df['news_count'] = np.random.poisson(5, len(df)) + (abs(price_df['returns'].fillna(0)) * 20).astype(int).values
        
        # Synthetic Google Trends (follows momentum with lag)
        momentum_20d = price_df['returns'].rolling(20).mean().fillna(0)
        df['search_interest'] = 50 + momentum_20d.values * 500 + np.random.normal(0, 10, len(df))
        df['search_interest'] = df['search_interest'].clip(0, 100)
        df['search_interest_zscore'] = (
            (df['search_interest'] - df['search_interest'].rolling(30).mean()) / 
            df['search_interest'].rolling(30).std()
        )
        
        return df


class AlternativeDataCollector:
    """
    Collects alternative data sources for enhanced signal generation:
    - Short interest data
    - Insider trading
    - Institutional holdings changes
    - Reddit/WSB mentions
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        
    def fetch_short_interest(self) -> pd.DataFrame:
        """Fetch short interest data (synthetic for demo)."""
        logger.info("Generating synthetic short interest data")
        
        # In production, would use FINRA/exchange data
        # Creating synthetic data for demonstration
        return pd.DataFrame()
    
    def generate_wsb_sentiment(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic WSB/Reddit sentiment.
        Retail interest typically spikes during rallies and high volatility.
        """
        df = pd.DataFrame(index=price_df.index)
        
        np.random.seed(43)
        
        # WSB mentions spike during volatility and positive momentum
        volatility = price_df['volatility_20d'].fillna(0.3)
        momentum = price_df['returns'].rolling(10).mean().fillna(0)
        
        base = np.random.poisson(20, len(price_df))
        wsb_spike = (volatility > volatility.rolling(60).mean()).astype(int) * 50
        momentum_spike = (momentum > 0.02).astype(int) * 30
        
        df['wsb_mentions'] = base + wsb_spike.values + momentum_spike.values
        df['wsb_sentiment'] = np.clip(momentum.values * 5 + np.random.normal(0, 0.2, len(df)), -1, 1)
        df['wsb_mentions_zscore'] = (
            (df['wsb_mentions'] - df['wsb_mentions'].rolling(30).mean()) / 
            df['wsb_mentions'].rolling(30).std()
        )
        
        return df


def create_comprehensive_dataset(
    ticker: str = "RKLB",
    start_date: str = "2021-08-25",
    end_date: Optional[str] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comprehensive dataset combining all data sources.
    """
    logger.info(f"Creating comprehensive dataset for {ticker}")
    
    # Collect market data
    market_collector = MarketDataCollector(ticker, start_date, end_date)
    market_data = market_collector.collect_all_data()
    
    # Collect sentiment data
    sentiment_collector = SentimentDataCollector(ticker)
    sentiment_data = sentiment_collector.generate_synthetic_sentiment(market_data)
    
    # Collect alternative data
    alt_collector = AlternativeDataCollector(ticker)
    wsb_data = alt_collector.generate_wsb_sentiment(market_data)
    
    # Merge all data
    comprehensive = market_data.join(sentiment_data, how='left')
    comprehensive = comprehensive.join(wsb_data, how='left')
    
    # Fill any remaining NaN values
    comprehensive = comprehensive.ffill().bfill()
    
    if save_path:
        comprehensive.to_csv(save_path)
        logger.info(f"Comprehensive dataset saved to {save_path}")
    
    logger.info(f"Comprehensive dataset created: {comprehensive.shape}")
    return comprehensive


if __name__ == "__main__":
    # Test data collection
    df = create_comprehensive_dataset(
        ticker="RKLB",
        start_date="2021-08-25",
        save_path="/workspace/rklb_anomaly_detection/data/rklb_comprehensive_data.csv"
    )
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
