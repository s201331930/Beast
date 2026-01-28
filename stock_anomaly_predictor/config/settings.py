"""
Configuration settings for Stock Anomaly Prediction System
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta


@dataclass
class DataConfig:
    """Data collection configuration"""
    ticker: str = "RKLB"
    start_date: str = "2021-08-25"  # RKLB IPO date
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    
    # Data sources
    use_yahoo_finance: bool = True
    use_google_trends: bool = True
    use_twitter: bool = True
    use_news: bool = True
    use_fred: bool = True  # Federal Reserve Economic Data
    
    # Market indicators to fetch
    market_tickers: List[str] = field(default_factory=lambda: [
        "^VIX",      # Volatility Index
        "^GSPC",     # S&P 500
        "^IXIC",     # NASDAQ
        "SPY",       # S&P 500 ETF
        "QQQ",       # NASDAQ ETF
        "XLE",       # Energy Select Sector
        "USO",       # Oil ETF
        "ARKX",      # Space ETF (competitor/sector)
        "IWM",       # Russell 2000 (small cap)
    ])
    
    # Related stocks for correlation analysis
    related_tickers: List[str] = field(default_factory=lambda: [
        "SPCE",      # Virgin Galactic
        "ASTR",      # Astra Space
        "RDW",       # Redwire
        "BKSY",      # BlackSky
        "PL",        # Planet Labs
        "ASTS",      # AST SpaceMobile
        "MNTS",      # Momentus
        "VORB",      # Virgin Orbit (if available)
        "BA",        # Boeing (aerospace)
        "LMT",       # Lockheed Martin
        "NOC",       # Northrop Grumman
        "RTX",       # Raytheon
    ])


@dataclass
class AnomalyConfig:
    """Anomaly detection configuration"""
    # Statistical thresholds
    zscore_threshold: float = 2.5
    iqr_multiplier: float = 1.5
    grubbs_alpha: float = 0.05
    
    # Volume anomaly
    volume_zscore_threshold: float = 2.0
    volume_ratio_threshold: float = 2.5  # vs 20-day average
    
    # Price movement thresholds
    big_move_threshold: float = 0.05  # 5% daily move
    rally_threshold: float = 0.10     # 10% move for rally detection
    
    # Lookback periods
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50
    extra_long_window: int = 200
    
    # ML model parameters
    isolation_forest_contamination: float = 0.05
    lof_neighbors: int = 20
    dbscan_eps: float = 0.5
    
    # Autoencoder
    autoencoder_threshold_percentile: float = 95


@dataclass
class TechnicalConfig:
    """Technical analysis configuration"""
    # Bollinger Bands
    bb_window: int = 20
    bb_std: float = 2.0
    
    # RSI
    rsi_window: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    
    # ATR
    atr_window: int = 14
    
    # Moving averages
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_windows: List[int] = field(default_factory=lambda: [9, 12, 26, 50])


@dataclass
class CyclicalConfig:
    """Cyclical and mean reversion configuration"""
    # Fourier analysis
    fourier_max_harmonics: int = 10
    cycle_detection_min_period: int = 5
    cycle_detection_max_period: int = 252  # 1 year trading days
    
    # Hurst exponent
    hurst_min_window: int = 10
    hurst_max_window: int = 100
    
    # Mean reversion
    half_life_max: int = 60
    
    # Regime detection
    regime_states: int = 3  # Low vol, normal, high vol


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    # Twitter/X
    twitter_lookback_days: int = 7
    min_tweet_volume: int = 100
    
    # Google Trends
    trends_timeframe: str = "today 3-m"
    trends_geo: str = "US"
    
    # News
    news_lookback_days: int = 3
    news_sources: List[str] = field(default_factory=lambda: [
        "reuters", "bloomberg", "wsj", "cnbc", "seekingalpha"
    ])
    
    # Sentiment thresholds
    bullish_threshold: float = 0.3
    bearish_threshold: float = -0.3


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # 10% per trade
    max_positions: int = 3
    
    # Signal thresholds
    entry_signal_threshold: float = 0.7  # Composite score
    exit_signal_threshold: float = 0.3
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.15  # 15%
    trailing_stop_pct: float = 0.08  # 8%
    max_holding_days: int = 30
    
    # Validation
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    walk_forward_windows: int = 5


@dataclass
class SignalWeights:
    """Weights for combining signals in ensemble"""
    # Anomaly detection weights
    statistical_anomaly: float = 0.15
    ml_anomaly: float = 0.15
    volume_anomaly: float = 0.12
    
    # Technical weights
    momentum: float = 0.12
    mean_reversion: float = 0.10
    trend_following: float = 0.08
    
    # Sentiment weights
    social_sentiment: float = 0.08
    news_sentiment: float = 0.08
    search_trends: float = 0.05
    
    # Market context weights
    vix_signal: float = 0.04
    sector_momentum: float = 0.03


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    cyclical: CyclicalConfig = field(default_factory=CyclicalConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    weights: SignalWeights = field(default_factory=SignalWeights)


# Global config instance
config = Config()
