"""
Sentiment Analysis Module
=========================
Analyzes sentiment from multiple sources for early rally detection:
- Social media sentiment (Twitter/X, Reddit)
- News sentiment
- Google Trends
- Options sentiment (put/call ratio anomalies)
- Fear & Greed indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    Multi-source sentiment analysis for detecting unusual interest/sentiment
    that may precede big stock moves.
    """
    
    def __init__(self, ticker: str = "RKLB"):
        self.ticker = ticker
        self.name = "Sentiment"
        
    def analyze_all_sentiment(self, price_data: pd.DataFrame, 
                              sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Comprehensive sentiment analysis combining multiple sources.
        """
        df = price_data.copy()
        
        # If no external sentiment data, generate synthetic indicators
        if sentiment_data is None:
            sentiment_data = self._generate_synthetic_sentiment(df)
        
        # Only add columns that don't already exist
        new_cols = [c for c in sentiment_data.columns if c not in df.columns]
        if new_cols:
            df = df.join(sentiment_data[new_cols], how='left')
        
        # Calculate sentiment anomalies
        df = self._calculate_sentiment_anomalies(df)
        
        # Calculate options sentiment
        df = self._calculate_options_sentiment(df)
        
        # Calculate market sentiment indicators
        df = self._calculate_market_sentiment(df)
        
        # Composite sentiment score
        df = self._calculate_composite_sentiment(df)
        
        return df
    
    def _generate_synthetic_sentiment(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic sentiment data based on price action patterns.
        This models how retail sentiment typically behaves.
        """
        df = pd.DataFrame(index=price_data.index)
        
        # Get returns for correlation
        if 'returns' in price_data.columns:
            returns = price_data['returns']
        else:
            returns = price_data['close'].pct_change()
        
        volatility = returns.rolling(20).std()
        momentum_5d = returns.rolling(5).mean()
        momentum_20d = returns.rolling(20).mean()
        
        np.random.seed(42)
        n = len(price_data)
        
        # Social Media Mentions (increases with volatility and positive momentum)
        base_mentions = 100 + np.random.poisson(50, n)
        vol_factor = (volatility.fillna(0.02) / 0.02).clip(0.5, 3) * 50
        momentum_factor = (momentum_5d.fillna(0) * 500).clip(-50, 100)
        
        df['social_mentions'] = base_mentions + vol_factor.values + momentum_factor.values
        df['social_mentions_ma7'] = df['social_mentions'].rolling(7).mean()
        df['social_mentions_ma30'] = df['social_mentions'].rolling(30).mean()
        df['social_mentions_zscore'] = (
            (df['social_mentions'] - df['social_mentions_ma30']) / 
            df['social_mentions'].rolling(30).std()
        )
        
        # Social Sentiment Score (-1 to 1, follows momentum with lag)
        noise = np.random.normal(0, 0.1, n)
        df['social_sentiment'] = np.clip(
            momentum_5d.fillna(0).values * 8 + 
            momentum_20d.fillna(0).values * 4 + 
            noise, -1, 1
        )
        df['social_sentiment_ma7'] = df['social_sentiment'].rolling(7).mean()
        
        # Reddit/WSB Mentions (spikes during high volatility periods)
        wsb_base = np.random.poisson(20, n)
        vol_spike = (volatility > volatility.rolling(60).mean() * 1.5).astype(int) * 80
        price_spike = (abs(returns) > 0.05).astype(int) * 50
        
        df['reddit_mentions'] = wsb_base + vol_spike.fillna(0).values + price_spike.fillna(0).values
        df['reddit_mentions_zscore'] = (
            (df['reddit_mentions'] - df['reddit_mentions'].rolling(30).mean()) /
            df['reddit_mentions'].rolling(30).std()
        )
        
        # Google Search Interest (0-100, follows price with lag)
        base_search = 30 + momentum_20d.fillna(0).values * 200 + np.random.normal(0, 10, n)
        df['search_interest'] = np.clip(base_search, 0, 100)
        df['search_interest_zscore'] = (
            (df['search_interest'] - df['search_interest'].rolling(30).mean()) /
            df['search_interest'].rolling(30).std()
        )
        
        # News Sentiment (-1 to 1)
        df['news_sentiment'] = np.clip(
            df['social_sentiment'] * 0.7 + np.random.normal(0, 0.2, n), -1, 1
        )
        df['news_count'] = np.random.poisson(3, n) + (abs(returns.fillna(0)) * 30).astype(int).values
        
        # Analyst Sentiment (changes less frequently)
        analyst_base = np.random.choice([-1, 0, 1], n, p=[0.2, 0.5, 0.3])
        df['analyst_sentiment'] = pd.Series(analyst_base, index=df.index).rolling(20, min_periods=1).mean()
        
        return df
    
    def _calculate_sentiment_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in sentiment metrics."""
        
        # Social Media Anomalies
        if 'social_mentions_zscore' in df.columns:
            df['social_mentions_anomaly'] = (np.abs(df['social_mentions_zscore']) > 2).astype(int)
            df['social_mentions_spike_up'] = (df['social_mentions_zscore'] > 2.5).astype(int)
        
        # Sentiment Shift (rapid change in sentiment)
        if 'social_sentiment' in df.columns:
            df['sentiment_shift'] = df['social_sentiment'].diff(5)
            df['sentiment_shift_anomaly'] = (np.abs(df['sentiment_shift']) > 0.5).astype(int)
        
        # Search Interest Anomalies
        if 'search_interest_zscore' in df.columns:
            df['search_anomaly'] = (df['search_interest_zscore'] > 2).astype(int)
        
        # Reddit/WSB Anomalies
        if 'reddit_mentions_zscore' in df.columns:
            df['reddit_anomaly'] = (df['reddit_mentions_zscore'] > 2.5).astype(int)
        
        # News Volume Anomaly
        if 'news_count' in df.columns:
            news_ma = df['news_count'].rolling(20).mean()
            news_std = df['news_count'].rolling(20).std()
            df['news_volume_zscore'] = (df['news_count'] - news_ma) / (news_std + 0.1)
            df['news_volume_anomaly'] = (df['news_volume_zscore'] > 2).astype(int)
        
        # Sentiment-Price Divergence
        if 'social_sentiment' in df.columns and 'returns' in df.columns:
            sentiment_ma = df['social_sentiment'].rolling(10).mean()
            returns_ma = df['returns'].rolling(10).mean()
            
            # Bullish divergence: sentiment up, price down
            df['sentiment_bullish_divergence'] = (
                (sentiment_ma > 0.3) & (returns_ma < -0.02)
            ).astype(int)
            
            # Bearish divergence: sentiment down, price up
            df['sentiment_bearish_divergence'] = (
                (sentiment_ma < -0.3) & (returns_ma > 0.02)
            ).astype(int)
        
        return df
    
    def _calculate_options_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate options-based sentiment indicators."""
        
        # Synthetic Put/Call Ratio (based on price/volatility behavior)
        if 'returns' in df.columns and 'volatility_20d' in df.columns:
            # Higher PCR during bearish periods
            bearish_indicator = (df['returns'].rolling(10).mean() < 0).astype(float)
            vol_factor = df['volatility_20d'].rank(pct=True)
            
            df['synthetic_pcr'] = 0.8 + bearish_indicator * 0.4 + vol_factor * 0.3 + np.random.normal(0, 0.1, len(df))
            df['synthetic_pcr'] = df['synthetic_pcr'].clip(0.3, 2.0)
            
            # PCR Anomalies
            pcr_ma = df['synthetic_pcr'].rolling(20).mean()
            pcr_std = df['synthetic_pcr'].rolling(20).std()
            df['pcr_zscore'] = (df['synthetic_pcr'] - pcr_ma) / (pcr_std + 0.01)
            
            # Extreme fear (high PCR) often precedes rallies
            df['pcr_extreme_fear'] = (df['pcr_zscore'] > 2).astype(int)
            # Extreme greed (low PCR) often precedes corrections
            df['pcr_extreme_greed'] = (df['pcr_zscore'] < -2).astype(int)
        
        # Implied Volatility Premium (synthetic)
        if 'volatility_20d' in df.columns:
            # IV typically trades at premium to realized vol
            iv_premium_base = 1.1 + np.random.normal(0, 0.1, len(df))
            df['iv_premium'] = iv_premium_base
            df['iv_premium_high'] = (df['iv_premium'] > 1.3).astype(int)
        
        return df
    
    def _calculate_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate broader market sentiment indicators."""
        
        # VIX-based Fear Index
        if 'vix_close' in df.columns:
            vix_ma = df['vix_close'].rolling(20).mean()
            df['vix_elevated'] = (df['vix_close'] > vix_ma * 1.2).astype(int)
            df['vix_spike'] = (df['vix_close'] > df['vix_close'].shift(1) * 1.15).astype(int)
            
            # VIX mean reversion signal (high VIX often precedes rallies)
            df['vix_reversion_signal'] = (
                (df['vix_close'] > vix_ma * 1.3) & 
                (df['vix_close'] < df['vix_close'].shift(1))
            ).astype(int)
        
        # Synthetic Fear & Greed Index (0-100)
        fear_greed_components = []
        
        if 'RSI' in df.columns:
            # RSI component
            rsi_score = df['RSI'].clip(0, 100)
            fear_greed_components.append(rsi_score)
        
        if 'volatility_20d' in df.columns:
            # Volatility component (inverse - high vol = fear)
            vol_percentile = df['volatility_20d'].rolling(252).rank(pct=True) * 100
            vol_score = 100 - vol_percentile
            fear_greed_components.append(vol_score)
        
        if 'synthetic_pcr' in df.columns:
            # PCR component (inverse - high PCR = fear)
            pcr_percentile = df['synthetic_pcr'].rolling(252).rank(pct=True) * 100
            pcr_score = 100 - pcr_percentile
            fear_greed_components.append(pcr_score)
        
        if fear_greed_components:
            df['fear_greed_index'] = pd.concat(fear_greed_components, axis=1).mean(axis=1)
            df['extreme_fear'] = (df['fear_greed_index'] < 25).astype(int)
            df['extreme_greed'] = (df['fear_greed_index'] > 75).astype(int)
        
        return df
    
    def _calculate_composite_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite sentiment score and signals."""
        
        # Collect all sentiment anomaly signals
        anomaly_cols = [
            'social_mentions_anomaly', 'sentiment_shift_anomaly', 
            'search_anomaly', 'reddit_anomaly', 'news_volume_anomaly',
            'pcr_extreme_fear', 'extreme_fear'
        ]
        available_anomalies = [c for c in anomaly_cols if c in df.columns]
        
        if available_anomalies:
            df['sentiment_anomaly_count'] = df[available_anomalies].sum(axis=1)
            df['sentiment_anomaly_signal'] = (df['sentiment_anomaly_count'] >= 2).astype(int)
        
        # Bullish sentiment signals (often precede rallies)
        bullish_cols = [
            'extreme_fear', 'pcr_extreme_fear', 'vix_reversion_signal',
            'sentiment_bullish_divergence'
        ]
        available_bullish = [c for c in bullish_cols if c in df.columns]
        
        if available_bullish:
            df['bullish_sentiment_count'] = df[available_bullish].sum(axis=1)
            df['bullish_sentiment_signal'] = (df['bullish_sentiment_count'] >= 2).astype(int)
        
        # Composite Sentiment Score (-1 to 1)
        sentiment_components = []
        
        if 'social_sentiment' in df.columns:
            sentiment_components.append(df['social_sentiment'])
        if 'news_sentiment' in df.columns:
            sentiment_components.append(df['news_sentiment'])
        if 'analyst_sentiment' in df.columns:
            sentiment_components.append(df['analyst_sentiment'])
        if 'fear_greed_index' in df.columns:
            # Normalize to -1 to 1
            fg_normalized = (df['fear_greed_index'] - 50) / 50
            sentiment_components.append(fg_normalized)
        
        if sentiment_components:
            df['composite_sentiment'] = pd.concat(sentiment_components, axis=1).mean(axis=1)
            df['composite_sentiment_bullish'] = (df['composite_sentiment'] > 0.3).astype(int)
            df['composite_sentiment_bearish'] = (df['composite_sentiment'] < -0.3).astype(int)
        
        # Early Warning Signals (combination of unusual activity)
        early_warning_cols = [
            'social_mentions_spike_up', 'search_anomaly', 'reddit_anomaly'
        ]
        available_warning = [c for c in early_warning_cols if c in df.columns]
        
        if available_warning:
            df['early_warning_signal'] = (df[available_warning].sum(axis=1) >= 1).astype(int)
        
        logger.info(f"Sentiment analysis complete: {len([c for c in df.columns if 'sentiment' in c.lower()])} sentiment features")
        
        return df


class VIXAnalyzer:
    """
    VIX-specific analysis for volatility regime detection.
    """
    
    def __init__(self):
        self.name = "VIX"
        
    def analyze(self, vix_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze VIX patterns for rally prediction."""
        
        df = price_data.copy()
        
        if 'vix_close' not in vix_data.columns:
            return df
        
        # Join VIX data
        vix = vix_data['vix_close'].reindex(df.index, method='ffill')
        
        # VIX Statistics
        df['vix'] = vix
        df['vix_ma10'] = vix.rolling(10).mean()
        df['vix_ma50'] = vix.rolling(50).mean()
        df['vix_std20'] = vix.rolling(20).std()
        
        # VIX Percentile (historical context)
        df['vix_percentile'] = vix.rolling(252).rank(pct=True) * 100
        
        # VIX Regime
        df['vix_regime'] = pd.cut(
            df['vix_percentile'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # VIX Term Structure Proxy (using MA relationship)
        df['vix_contango'] = (df['vix_ma50'] > df['vix']).astype(int)  # Backwardation
        df['vix_backwardation'] = (df['vix_ma50'] < df['vix']).astype(int)  # Contango
        
        # VIX Spike Detection
        df['vix_spike'] = (vix > df['vix_ma10'] * 1.25).astype(int)
        df['vix_spike_reversal'] = (
            (df['vix_spike'].shift(1) == 1) & 
            (vix < vix.shift(1))
        ).astype(int)
        
        # VIX Crush (rapid decline often follows rallies)
        df['vix_crush'] = (vix.pct_change(5) < -0.15).astype(int)
        
        # VIX-Price Correlation
        df['vix_price_corr'] = vix.rolling(20).corr(df['close'].pct_change().rolling(20).sum())
        
        # VIX Mean Reversion Signal
        df['vix_mean_reversion'] = (
            (df['vix_percentile'] > 80) & 
            (vix < vix.shift(1)) &
            (vix < vix.shift(2))
        ).astype(int)
        
        return df


if __name__ == "__main__":
    import yfinance as yf
    
    # Test sentiment analysis
    rklb = yf.Ticker("RKLB").history(period="2y")
    rklb.columns = [c.lower() for c in rklb.columns]
    rklb['returns'] = rklb['close'].pct_change()
    rklb['volatility_20d'] = rklb['returns'].rolling(20).std() * np.sqrt(252)
    
    analyzer = SentimentAnalyzer("RKLB")
    result = analyzer.analyze_all_sentiment(rklb)
    
    print("Sentiment Analysis Results:")
    print(result.tail(10))
    
    # Check for signals
    if 'bullish_sentiment_signal' in result.columns:
        bullish_days = result[result['bullish_sentiment_signal'] == 1]
        print(f"\nBullish sentiment signals: {len(bullish_days)}")
