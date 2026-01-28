"""
Comprehensive Data Collection Module
Fetches price, volume, sentiment, and market data from multiple sources
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class DataCollector:
    """
    Multi-source data collector for stock anomaly prediction.
    Aggregates data from various financial and sentiment sources.
    """
    
    def __init__(self, ticker: str = None):
        self.ticker = ticker or config.data.ticker
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        self.data = {}
        
    def fetch_price_data(self, ticker: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance with extended metrics.
        """
        ticker = ticker or self.ticker
        print(f"Fetching price data for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                print(f"Warning: No data for {ticker}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Add basic derived features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_range'] = (df['close'] - df['open']) / df['open']
            df['volume_change'] = df['volume'].pct_change()
            
            # Dollar volume
            df['dollar_volume'] = df['close'] * df['volume']
            
            # Gap analysis
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            
            # Intraday volatility
            df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
            
            # True Range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            print(f"  Fetched {len(df)} days of data for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_options_data(self, ticker: str = None) -> pd.DataFrame:
        """
        Fetch options data for put/call ratio analysis.
        """
        ticker = ticker or self.ticker
        print(f"Fetching options data for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get all expiration dates
            expirations = stock.options
            
            if not expirations:
                print(f"  No options data available for {ticker}")
                return pd.DataFrame()
            
            options_data = []
            
            # Get options for nearest expirations (focus on short-term)
            for exp_date in expirations[:3]:  # First 3 expirations
                try:
                    opt = stock.option_chain(exp_date)
                    calls = opt.calls
                    puts = opt.puts
                    
                    # Calculate metrics
                    total_call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
                    total_put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
                    total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
                    total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
                    
                    # Put/Call ratios
                    pc_volume_ratio = total_put_volume / max(total_call_volume, 1)
                    pc_oi_ratio = total_put_oi / max(total_call_oi, 1)
                    
                    options_data.append({
                        'expiration': exp_date,
                        'call_volume': total_call_volume,
                        'put_volume': total_put_volume,
                        'call_oi': total_call_oi,
                        'put_oi': total_put_oi,
                        'pc_volume_ratio': pc_volume_ratio,
                        'pc_oi_ratio': pc_oi_ratio
                    })
                except:
                    continue
            
            if options_data:
                df = pd.DataFrame(options_data)
                print(f"  Fetched options data for {len(options_data)} expirations")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return pd.DataFrame()
    
    def fetch_market_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market indicators: VIX, S&P 500, sector ETFs, etc.
        """
        print("Fetching market indicators...")
        market_data = {}
        
        for ticker in config.data.market_tickers:
            try:
                df = self.fetch_price_data(ticker)
                if not df.empty:
                    market_data[ticker] = df
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                
        return market_data
    
    def fetch_related_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch related stocks for sector/peer analysis.
        """
        print("Fetching related stocks...")
        related_data = {}
        
        for ticker in config.data.related_tickers:
            try:
                df = self.fetch_price_data(ticker)
                if not df.empty:
                    related_data[ticker] = df
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                
        return related_data
    
    def fetch_google_trends(self, keywords: List[str] = None) -> pd.DataFrame:
        """
        Fetch Google Trends data for sentiment analysis.
        """
        if keywords is None:
            keywords = [
                self.ticker,
                "Rocket Lab",
                "RKLB stock",
                "Electron rocket",
                "Peter Beck"
            ]
        
        print(f"Fetching Google Trends for: {keywords}")
        
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Fetch interest over time
            pytrends.build_payload(
                keywords[:5],  # Max 5 keywords
                timeframe=f"{self.start_date} {self.end_date}",
                geo=config.sentiment.trends_geo
            )
            
            df = pytrends.interest_over_time()
            
            if not df.empty:
                # Drop isPartial column if exists
                if 'isPartial' in df.columns:
                    df = df.drop('isPartial', axis=1)
                    
                # Create aggregate search interest
                df['aggregate_interest'] = df.mean(axis=1)
                df['interest_change'] = df['aggregate_interest'].pct_change()
                
                print(f"  Fetched {len(df)} days of trends data")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        """
        if df.empty:
            return df
            
        print("Calculating technical indicators...")
        
        # Moving Averages
        for window in config.technical.sma_windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'sma_{window}_slope'] = df[f'sma_{window}'].pct_change(5)
            
        for window in config.technical.ema_windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Bollinger Bands
        bb_window = config.technical.bb_window
        bb_std = config.technical.bb_std
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_rolling_std = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * bb_rolling_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * bb_rolling_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        rsi_window = config.technical.rsi_window
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_divergence'] = df['rsi'].diff(5) - df['close'].pct_change(5) * 100
        
        # MACD
        exp1 = df['close'].ewm(span=config.technical.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=config.technical.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=config.technical.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_change'] = df['macd_histogram'].diff()
        
        # Stochastic Oscillator
        stoch_k = config.technical.stoch_k
        stoch_d = config.technical.stoch_d
        low_min = df['low'].rolling(window=stoch_k).min()
        high_max = df['high'].rolling(window=stoch_k).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=stoch_d).mean()
        
        # ATR (Average True Range)
        atr_window = config.technical.atr_window
        df['atr'] = df['true_range'].rolling(window=atr_window).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_divergence'] = df['obv'].pct_change(10) - df['close'].pct_change(10)
        
        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        mf_ratio = money_flow_positive.rolling(14).sum() / money_flow_negative.rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        
        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['ad'] = (clv * df['volume']).fillna(0).cumsum()
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # Williams %R
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Price relative to moving averages
        for window in [20, 50, 200]:
            if f'sma_{window}' in df.columns:
                df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Volatility measures
        df['volatility_10'] = df['returns'].rolling(window=10).std() * np.sqrt(252)
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        
        # Historical volatility percentile
        df['vol_percentile'] = df['volatility_20'].rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 20 else np.nan
        )
        
        print(f"  Calculated {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} indicators")
        return df
    
    def collect_all_data(self) -> Dict[str, any]:
        """
        Collect all data from various sources and compile into a comprehensive dataset.
        """
        print("=" * 60)
        print(f"Starting comprehensive data collection for {self.ticker}")
        print("=" * 60)
        
        # Primary stock data
        primary_df = self.fetch_price_data()
        if primary_df.empty:
            raise ValueError(f"Failed to fetch primary data for {self.ticker}")
        
        # Add technical indicators
        primary_df = self.calculate_technical_indicators(primary_df)
        
        # Fetch market indicators
        market_data = self.fetch_market_indicators()
        
        # Fetch related stocks
        related_data = self.fetch_related_stocks()
        
        # Fetch options data (for current P/C ratio snapshot)
        options_data = self.fetch_options_data()
        
        # Try to fetch Google Trends
        try:
            trends_data = self.fetch_google_trends()
        except:
            trends_data = pd.DataFrame()
        
        # Merge VIX data if available
        if "^VIX" in market_data:
            vix_df = market_data["^VIX"][['close']].rename(columns={'close': 'vix'})
            primary_df = primary_df.join(vix_df, how='left')
            primary_df['vix'] = primary_df['vix'].ffill()
            
            # VIX metrics
            primary_df['vix_sma_20'] = primary_df['vix'].rolling(window=20).mean()
            primary_df['vix_ratio'] = primary_df['vix'] / primary_df['vix_sma_20']
            primary_df['vix_percentile'] = primary_df['vix'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 20 else np.nan
            )
        
        # Calculate sector correlation if space ETF available
        if "ARKX" in market_data:
            arkx_returns = market_data["ARKX"]['returns'].rename('arkx_returns')
            primary_df = primary_df.join(arkx_returns, how='left')
            primary_df['sector_correlation'] = primary_df['returns'].rolling(window=20).corr(primary_df['arkx_returns'])
        
        # Calculate market beta
        if "^GSPC" in market_data:
            sp500_returns = market_data["^GSPC"]['returns'].rename('sp500_returns')
            primary_df = primary_df.join(sp500_returns, how='left')
            
            # Rolling beta
            def calc_beta(df_window):
                if len(df_window) < 20:
                    return np.nan
                cov = df_window['returns'].cov(df_window['sp500_returns'])
                var = df_window['sp500_returns'].var()
                return cov / var if var > 0 else np.nan
            
            primary_df['market_beta'] = primary_df[['returns', 'sp500_returns']].rolling(window=60).apply(
                lambda x: calc_beta(pd.DataFrame({'returns': x[:len(x)//2], 'sp500_returns': x[len(x)//2:]})),
                raw=False
            )
        
        # Calculate relative strength vs peers
        peer_returns = []
        for ticker, df in related_data.items():
            if not df.empty and 'returns' in df.columns:
                peer_returns.append(df['returns'].rename(f'{ticker}_returns'))
        
        if peer_returns:
            peer_df = pd.concat(peer_returns, axis=1)
            peer_df['peer_avg_returns'] = peer_df.mean(axis=1)
            primary_df = primary_df.join(peer_df['peer_avg_returns'], how='left')
            primary_df['relative_strength'] = primary_df['returns'] - primary_df['peer_avg_returns']
            primary_df['rs_cumsum'] = primary_df['relative_strength'].cumsum()
        
        # Merge Google Trends if available
        if not trends_data.empty:
            primary_df = primary_df.join(trends_data[['aggregate_interest', 'interest_change']], how='left')
            primary_df['aggregate_interest'] = primary_df['aggregate_interest'].ffill()
            primary_df['interest_change'] = primary_df['interest_change'].ffill()
        
        # Store all data
        self.data = {
            'primary': primary_df,
            'market': market_data,
            'related': related_data,
            'options': options_data,
            'trends': trends_data
        }
        
        print("=" * 60)
        print(f"Data collection complete!")
        print(f"  Primary data: {len(primary_df)} rows, {len(primary_df.columns)} columns")
        print(f"  Market indicators: {len(market_data)} tickers")
        print(f"  Related stocks: {len(related_data)} tickers")
        print("=" * 60)
        
        return self.data
    
    def get_primary_dataframe(self) -> pd.DataFrame:
        """Return the primary enriched dataframe."""
        if 'primary' in self.data:
            return self.data['primary']
        return pd.DataFrame()
    
    def save_data(self, path: str = "data/collected_data.pkl"):
        """Save collected data to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Data saved to {path}")
    
    def load_data(self, path: str = "data/collected_data.pkl"):
        """Load previously collected data."""
        import pickle
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Data loaded from {path}")
        return self.data


if __name__ == "__main__":
    # Test data collection
    collector = DataCollector("RKLB")
    data = collector.collect_all_data()
    
    # Display summary
    primary = data['primary']
    print("\nPrimary DataFrame Summary:")
    print(primary.tail())
    print("\nColumns:", primary.columns.tolist())
