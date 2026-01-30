"""
Stock Suitability Screener Module

Scientific pre-screening to identify stocks where the anomaly prediction
system is most effective, based on empirical findings:

1. MOMENTUM REGIME: High-momentum stocks with strong trends
2. RETAIL INTEREST: Stocks with retail/sentiment drivers
3. SYSTEMATIC RISK: High beta stocks respond better to signals
4. MARKET REGIME: Bull market conditions enhance predictability
5. VOLATILITY PROFILE: Sufficient volatility for profitable trades
6. LIQUIDITY: Adequate volume for execution

Mathematical Foundation:
- Momentum: Rate of change, trend strength metrics
- Hurst Exponent: Trending vs mean-reverting behavior
- Beta: CAPM sensitivity to market movements
- ADX: Trend strength indicator
- Retail Score: Options activity, social volume proxies

Empirical Evidence (from backtests):
- RKLB (beta=4.52, Hurst=1.04): 69.6% win rate, profit factor 1.96
- NVDA (beta=1.89, Hurst=1.02): 65.1% win rate, profit factor 1.57
- BABA (beta=0.69, Hurst=1.00): 51.9% win rate, profit factor 1.04

Conclusion: Higher beta + higher Hurst = better system performance
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StockSuitabilityScore:
    """Comprehensive suitability assessment for a stock."""
    ticker: str
    overall_score: float  # 0-100
    recommendation: str   # 'EXCELLENT', 'GOOD', 'MODERATE', 'POOR'
    
    # Component scores (0-100)
    momentum_score: float
    trend_strength_score: float
    beta_score: float
    volatility_score: float
    liquidity_score: float
    retail_interest_score: float
    market_regime_score: float
    
    # Raw metrics
    hurst_exponent: float
    beta: float
    adx: float
    avg_volume: float
    volatility: float
    momentum_20d: float
    
    # Flags
    is_trending: bool
    is_high_beta: bool
    has_retail_interest: bool
    in_bull_regime: bool
    
    def __str__(self):
        return f"""
Stock Suitability Analysis: {self.ticker}
{'='*50}
Overall Score: {self.overall_score:.1f}/100 ({self.recommendation})

Component Scores:
  Momentum:        {self.momentum_score:.1f}/100
  Trend Strength:  {self.trend_strength_score:.1f}/100
  Beta:            {self.beta_score:.1f}/100
  Volatility:      {self.volatility_score:.1f}/100
  Liquidity:       {self.liquidity_score:.1f}/100
  Retail Interest: {self.retail_interest_score:.1f}/100
  Market Regime:   {self.market_regime_score:.1f}/100

Key Metrics:
  Hurst Exponent:  {self.hurst_exponent:.3f} ({'Trending' if self.is_trending else 'Mean Reverting'})
  Beta:            {self.beta:.2f} ({'High' if self.is_high_beta else 'Low'})
  ADX:             {self.adx:.1f}
  20-Day Momentum: {self.momentum_20d:+.1%}
  Volatility:      {self.volatility:.1%}

Flags:
  ✓ Trending:       {'Yes' if self.is_trending else 'No'}
  ✓ High Beta:      {'Yes' if self.is_high_beta else 'No'}
  ✓ Retail Interest:{'Yes' if self.has_retail_interest else 'No'}
  ✓ Bull Regime:    {'Yes' if self.in_bull_regime else 'No'}
"""


class StockSuitabilityScreener:
    """
    Scientific screener to determine if a stock is suitable for
    the anomaly prediction system.
    
    Based on empirical evidence from RKLB, NVDA, and BABA backtests.
    """
    
    # Empirically derived thresholds from backtest results
    OPTIMAL_HURST_MIN = 0.55       # Trending behavior threshold
    OPTIMAL_BETA_MIN = 1.2         # High beta threshold
    OPTIMAL_ADX_MIN = 20           # Trend strength threshold
    OPTIMAL_VOLATILITY_MIN = 0.30  # 30% annual volatility minimum
    OPTIMAL_VOLATILITY_MAX = 1.50  # 150% annual volatility maximum
    
    # Weights for overall score (empirically tuned)
    WEIGHTS = {
        'momentum': 0.20,
        'trend_strength': 0.20,
        'beta': 0.20,
        'volatility': 0.15,
        'liquidity': 0.10,
        'retail_interest': 0.10,
        'market_regime': 0.05
    }
    
    def __init__(self, df: pd.DataFrame, market_df: pd.DataFrame = None):
        """
        Initialize screener with stock data.
        
        Args:
            df: Stock price DataFrame with OHLCV data
            market_df: Optional market index DataFrame for beta calculation
        """
        self.df = df.copy()
        self.market_df = market_df
        self.metrics = {}
        
    def calculate_hurst_exponent(self, series: pd.Series, 
                                  min_window: int = 10,
                                  max_window: int = 100) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H > 0.5: Trending (persistent) - GOOD for system
        H < 0.5: Mean reverting (anti-persistent) - POOR for system
        H = 0.5: Random walk
        
        Returns:
            Hurst exponent value
        """
        series = series.dropna().values
        n = len(series)
        
        if n < max_window:
            max_window = n // 2
        
        rs_list = []
        n_list = []
        
        for window in range(min_window, max_window + 1, 5):
            rs_values = []
            
            for start in range(0, n - window, window // 2):
                subset = series[start:start + window]
                
                mean_adj = subset - np.mean(subset)
                cumsum = np.cumsum(mean_adj)
                
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subset, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                rs_list.append(np.mean(rs_values))
                n_list.append(window)
        
        if len(rs_list) > 2:
            log_n = np.log(n_list)
            log_rs = np.log(rs_list)
            slope, _ = np.polyfit(log_n, log_rs, 1)
            return slope
        
        return 0.5
    
    def calculate_beta(self, stock_returns: pd.Series, 
                       market_returns: pd.Series,
                       window: int = 252) -> float:
        """
        Calculate rolling beta using CAPM regression.
        
        β = Cov(R_stock, R_market) / Var(R_market)
        
        High beta stocks (β > 1.2) have shown better system performance.
        
        Returns:
            Beta coefficient
        """
        # Align series
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 60:
            return 1.0
        
        # Use recent data
        aligned = aligned.tail(window)
        
        stock = aligned.iloc[:, 0]
        market = aligned.iloc[:, 1]
        
        cov = stock.cov(market)
        var = market.var()
        
        if var > 0:
            return cov / var
        return 1.0
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        
        ADX > 25: Strong trend - GOOD for system
        ADX < 20: Weak/no trend - POOR for system
        
        Returns:
            Current ADX value
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.iloc[-1] if not adx.empty else 0
    
    def calculate_momentum_metrics(self) -> Dict:
        """
        Calculate comprehensive momentum metrics.
        
        Returns:
            Dictionary of momentum metrics
        """
        close = self.df['close']
        
        # Price momentum (various timeframes)
        momentum_5d = close.pct_change(5).iloc[-1]
        momentum_10d = close.pct_change(10).iloc[-1]
        momentum_20d = close.pct_change(20).iloc[-1]
        momentum_60d = close.pct_change(60).iloc[-1]
        
        # Rate of change acceleration
        roc_10 = close.pct_change(10)
        roc_acceleration = roc_10.diff(5).iloc[-1]
        
        # Price vs moving averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        
        price = close.iloc[-1]
        
        above_sma_20 = price > sma_20
        above_sma_50 = price > sma_50
        above_sma_200 = price > sma_200
        
        # Moving average alignment (bullish when 20 > 50 > 200)
        ma_alignment = (sma_20 > sma_50) and (sma_50 > sma_200)
        
        # Composite momentum score
        momentum_factors = [
            momentum_20d > 0,
            momentum_60d > 0,
            above_sma_20,
            above_sma_50,
            above_sma_200,
            ma_alignment,
            roc_acceleration > 0
        ]
        
        momentum_score = sum(momentum_factors) / len(momentum_factors) * 100
        
        return {
            'momentum_5d': momentum_5d,
            'momentum_10d': momentum_10d,
            'momentum_20d': momentum_20d,
            'momentum_60d': momentum_60d,
            'roc_acceleration': roc_acceleration,
            'above_sma_20': above_sma_20,
            'above_sma_50': above_sma_50,
            'above_sma_200': above_sma_200,
            'ma_alignment': ma_alignment,
            'momentum_score': momentum_score
        }
    
    def calculate_volatility_metrics(self) -> Dict:
        """
        Calculate volatility metrics.
        
        Optimal volatility: 30-150% annualized
        Too low: Not enough movement for profitable trades
        Too high: Excessive noise/risk
        
        Returns:
            Dictionary of volatility metrics
        """
        returns = self.df['close'].pct_change().dropna()
        
        # Annualized volatility
        volatility_20d = returns.tail(20).std() * np.sqrt(252)
        volatility_60d = returns.tail(60).std() * np.sqrt(252)
        volatility_252d = returns.tail(252).std() * np.sqrt(252)
        
        # Volatility regime
        vol_percentile = stats.percentileofscore(
            returns.rolling(20).std().dropna() * np.sqrt(252),
            volatility_20d
        ) / 100
        
        # ATR-based volatility
        if 'high' in self.df.columns and 'low' in self.df.columns:
            tr = np.maximum(
                self.df['high'] - self.df['low'],
                np.maximum(
                    abs(self.df['high'] - self.df['close'].shift(1)),
                    abs(self.df['low'] - self.df['close'].shift(1))
                )
            )
            atr_pct = (tr.rolling(14).mean() / self.df['close']).iloc[-1]
        else:
            atr_pct = volatility_20d / np.sqrt(252)
        
        # Score volatility (optimal range: 30-100%)
        if volatility_60d < self.OPTIMAL_VOLATILITY_MIN:
            vol_score = volatility_60d / self.OPTIMAL_VOLATILITY_MIN * 50
        elif volatility_60d > self.OPTIMAL_VOLATILITY_MAX:
            vol_score = max(0, 100 - (volatility_60d - self.OPTIMAL_VOLATILITY_MAX) * 100)
        else:
            # Optimal range
            vol_score = 70 + (volatility_60d - 0.30) / 0.70 * 30
        
        return {
            'volatility_20d': volatility_20d,
            'volatility_60d': volatility_60d,
            'volatility_252d': volatility_252d,
            'vol_percentile': vol_percentile,
            'atr_pct': atr_pct,
            'volatility_score': min(100, max(0, vol_score))
        }
    
    def calculate_liquidity_metrics(self) -> Dict:
        """
        Calculate liquidity metrics.
        
        Adequate liquidity ensures:
        - Reliable price discovery for signals
        - Execution without significant slippage
        
        Returns:
            Dictionary of liquidity metrics
        """
        volume = self.df['volume']
        close = self.df['close']
        
        # Average daily volume
        avg_volume_20d = volume.tail(20).mean()
        avg_volume_60d = volume.tail(60).mean()
        
        # Dollar volume
        dollar_volume = (close * volume).tail(20).mean()
        
        # Volume trend
        vol_sma_short = volume.rolling(10).mean().iloc[-1]
        vol_sma_long = volume.rolling(50).mean().iloc[-1]
        volume_trend = vol_sma_short / vol_sma_long if vol_sma_long > 0 else 1
        
        # Liquidity score (based on dollar volume)
        # Minimum $10M daily for good liquidity
        if dollar_volume >= 100_000_000:  # $100M+
            liquidity_score = 100
        elif dollar_volume >= 50_000_000:  # $50M+
            liquidity_score = 90
        elif dollar_volume >= 10_000_000:  # $10M+
            liquidity_score = 70
        elif dollar_volume >= 1_000_000:   # $1M+
            liquidity_score = 50
        else:
            liquidity_score = dollar_volume / 1_000_000 * 50
        
        return {
            'avg_volume_20d': avg_volume_20d,
            'avg_volume_60d': avg_volume_60d,
            'dollar_volume': dollar_volume,
            'volume_trend': volume_trend,
            'liquidity_score': liquidity_score
        }
    
    def calculate_retail_interest_score(self) -> Dict:
        """
        Estimate retail interest based on available proxies.
        
        High retail interest indicators:
        - High options volume relative to stock volume
        - Price < $200 (more accessible)
        - High volume spikes (retail FOMO)
        - Recent IPO or momentum stock characteristics
        
        Returns:
            Dictionary with retail interest metrics
        """
        close = self.df['close']
        volume = self.df['volume']
        
        current_price = close.iloc[-1]
        
        # Price accessibility (retail prefers lower prices)
        if current_price < 20:
            price_access_score = 100
        elif current_price < 50:
            price_access_score = 85
        elif current_price < 100:
            price_access_score = 70
        elif current_price < 200:
            price_access_score = 55
        else:
            price_access_score = 40
        
        # Volume spike frequency (retail FOMO indicator)
        volume_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
        spike_frequency = (volume_zscore > 2).tail(60).sum() / 60 * 100
        
        # Volatility of volume (retail creates erratic volume)
        volume_volatility = volume.pct_change().tail(60).std()
        vol_vol_score = min(100, volume_volatility * 500)
        
        # Recent momentum (retail chases momentum)
        recent_momentum = close.pct_change(20).iloc[-1]
        momentum_chase_score = min(100, max(0, 50 + recent_momentum * 200))
        
        # Composite retail score
        retail_score = (
            price_access_score * 0.25 +
            spike_frequency * 0.25 +
            vol_vol_score * 0.25 +
            momentum_chase_score * 0.25
        )
        
        return {
            'price_accessibility': price_access_score,
            'volume_spike_frequency': spike_frequency,
            'volume_volatility_score': vol_vol_score,
            'momentum_chase_score': momentum_chase_score,
            'retail_interest_score': retail_score
        }
    
    def calculate_market_regime_score(self) -> Dict:
        """
        Assess current market regime.
        
        System performs better in:
        - Bull markets (rising prices)
        - Low VIX environments
        - Risk-on conditions
        
        Returns:
            Dictionary with market regime metrics
        """
        close = self.df['close']
        
        # Stock's own regime
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        
        # Golden/Death cross
        golden_cross = sma_50 > sma_200
        
        # Distance from 200 SMA
        price = close.iloc[-1]
        distance_200sma = (price - sma_200) / sma_200
        
        # Recent drawdown
        rolling_max = close.rolling(252).max()
        current_drawdown = (close.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
        
        # Regime score
        regime_factors = []
        
        # Golden cross is bullish
        regime_factors.append(100 if golden_cross else 30)
        
        # Price above 200 SMA
        regime_factors.append(min(100, max(0, 50 + distance_200sma * 100)))
        
        # Low drawdown is good
        regime_factors.append(min(100, max(0, 100 + current_drawdown * 200)))
        
        regime_score = np.mean(regime_factors)
        
        return {
            'golden_cross': golden_cross,
            'distance_200sma': distance_200sma,
            'current_drawdown': current_drawdown,
            'in_bull_regime': golden_cross and distance_200sma > 0,
            'market_regime_score': regime_score
        }
    
    def calculate_trend_strength_score(self) -> Dict:
        """
        Calculate comprehensive trend strength metrics.
        
        Strong trends (ADX > 25, Hurst > 0.55) are optimal for the system.
        
        Returns:
            Dictionary with trend metrics
        """
        close = self.df['close']
        high = self.df.get('high', close)
        low = self.df.get('low', close)
        
        # Hurst exponent
        hurst = self.calculate_hurst_exponent(close)
        
        # ADX
        adx = self.calculate_adx(high, low, close)
        
        # Linear regression R-squared (trend fit)
        x = np.arange(len(close.tail(60)))
        y = close.tail(60).values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Trend direction
        trend_direction = 'UP' if slope > 0 else 'DOWN'
        
        # Score components
        hurst_score = min(100, max(0, (hurst - 0.3) / 0.4 * 100))
        adx_score = min(100, max(0, adx / 50 * 100))
        r2_score = r_squared * 100
        
        # Combined trend strength score
        trend_score = (
            hurst_score * 0.40 +
            adx_score * 0.35 +
            r2_score * 0.25
        )
        
        return {
            'hurst_exponent': hurst,
            'adx': adx,
            'r_squared': r_squared,
            'trend_direction': trend_direction,
            'trend_slope': slope,
            'is_trending': hurst > self.OPTIMAL_HURST_MIN and adx > self.OPTIMAL_ADX_MIN,
            'hurst_score': hurst_score,
            'adx_score': adx_score,
            'trend_strength_score': trend_score
        }
    
    def calculate_beta_score(self, market_returns: pd.Series = None) -> Dict:
        """
        Calculate beta and related metrics.
        
        High beta (> 1.2) stocks have shown better system performance.
        
        Args:
            market_returns: Market index returns (optional)
            
        Returns:
            Dictionary with beta metrics
        """
        stock_returns = self.df['close'].pct_change().dropna()
        
        if market_returns is not None:
            beta = self.calculate_beta(stock_returns, market_returns)
        elif self.market_df is not None and 'close' in self.market_df.columns:
            market_returns = self.market_df['close'].pct_change().dropna()
            beta = self.calculate_beta(stock_returns, market_returns)
        else:
            # Estimate beta from volatility ratio (proxy)
            stock_vol = stock_returns.std()
            beta = stock_vol / 0.01  # Assume 1% daily market vol
            beta = min(beta, 4.0)  # Cap at 4
        
        # Score beta
        # Optimal range: 1.2 - 3.0
        if beta < 0.5:
            beta_score = beta / 0.5 * 30
        elif beta < self.OPTIMAL_BETA_MIN:
            beta_score = 30 + (beta - 0.5) / 0.7 * 30
        elif beta <= 3.0:
            beta_score = 60 + (beta - 1.2) / 1.8 * 40
        else:
            beta_score = 100 - (beta - 3.0) * 10  # Penalize extreme beta
        
        beta_score = min(100, max(0, beta_score))
        
        return {
            'beta': beta,
            'is_high_beta': beta >= self.OPTIMAL_BETA_MIN,
            'beta_score': beta_score
        }
    
    def analyze(self, market_returns: pd.Series = None) -> StockSuitabilityScore:
        """
        Run complete suitability analysis.
        
        Args:
            market_returns: Optional market index returns
            
        Returns:
            StockSuitabilityScore with comprehensive assessment
        """
        print("Analyzing stock suitability...")
        
        # Calculate all metrics
        momentum = self.calculate_momentum_metrics()
        trend = self.calculate_trend_strength_score()
        beta_metrics = self.calculate_beta_score(market_returns)
        volatility = self.calculate_volatility_metrics()
        liquidity = self.calculate_liquidity_metrics()
        retail = self.calculate_retail_interest_score()
        regime = self.calculate_market_regime_score()
        
        # Store all metrics
        self.metrics = {
            'momentum': momentum,
            'trend': trend,
            'beta': beta_metrics,
            'volatility': volatility,
            'liquidity': liquidity,
            'retail': retail,
            'regime': regime
        }
        
        # Calculate overall score
        overall_score = (
            self.WEIGHTS['momentum'] * momentum['momentum_score'] +
            self.WEIGHTS['trend_strength'] * trend['trend_strength_score'] +
            self.WEIGHTS['beta'] * beta_metrics['beta_score'] +
            self.WEIGHTS['volatility'] * volatility['volatility_score'] +
            self.WEIGHTS['liquidity'] * liquidity['liquidity_score'] +
            self.WEIGHTS['retail_interest'] * retail['retail_interest_score'] +
            self.WEIGHTS['market_regime'] * regime['market_regime_score']
        )
        
        # Determine recommendation
        if overall_score >= 75:
            recommendation = 'EXCELLENT'
        elif overall_score >= 60:
            recommendation = 'GOOD'
        elif overall_score >= 45:
            recommendation = 'MODERATE'
        else:
            recommendation = 'POOR'
        
        # Create result object
        result = StockSuitabilityScore(
            ticker=self.df.name if hasattr(self.df, 'name') else 'UNKNOWN',
            overall_score=overall_score,
            recommendation=recommendation,
            
            momentum_score=momentum['momentum_score'],
            trend_strength_score=trend['trend_strength_score'],
            beta_score=beta_metrics['beta_score'],
            volatility_score=volatility['volatility_score'],
            liquidity_score=liquidity['liquidity_score'],
            retail_interest_score=retail['retail_interest_score'],
            market_regime_score=regime['market_regime_score'],
            
            hurst_exponent=trend['hurst_exponent'],
            beta=beta_metrics['beta'],
            adx=trend['adx'],
            avg_volume=liquidity['avg_volume_20d'],
            volatility=volatility['volatility_60d'],
            momentum_20d=momentum['momentum_20d'],
            
            is_trending=trend['is_trending'],
            is_high_beta=beta_metrics['is_high_beta'],
            has_retail_interest=retail['retail_interest_score'] > 60,
            in_bull_regime=regime['in_bull_regime']
        )
        
        return result


class MultiStockScreener:
    """
    Screen multiple stocks to find best candidates for the anomaly system.
    """
    
    def __init__(self):
        self.results = {}
        
    def screen_stocks(self, 
                      tickers: List[str],
                      start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
        """
        Screen multiple stocks and rank by suitability.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with ranked stocks
        """
        import yfinance as yf
        
        print(f"Screening {len(tickers)} stocks...")
        
        results = []
        
        # Get market data for beta calculation
        market = yf.Ticker("^GSPC")
        market_df = market.history(start=start_date, end=end_date)
        market_returns = market_df['Close'].pct_change().dropna()
        
        for ticker in tickers:
            try:
                print(f"  Analyzing {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if len(df) < 100:
                    print(f"    Insufficient data for {ticker}")
                    continue
                
                df.columns = [c.lower() for c in df.columns]
                df.name = ticker
                
                screener = StockSuitabilityScreener(df, market_df)
                score = screener.analyze(market_returns)
                
                results.append({
                    'ticker': ticker,
                    'overall_score': score.overall_score,
                    'recommendation': score.recommendation,
                    'momentum_score': score.momentum_score,
                    'trend_score': score.trend_strength_score,
                    'beta_score': score.beta_score,
                    'beta': score.beta,
                    'hurst': score.hurst_exponent,
                    'adx': score.adx,
                    'volatility': score.volatility,
                    'is_trending': score.is_trending,
                    'is_high_beta': score.is_high_beta,
                    'has_retail_interest': score.has_retail_interest,
                    'in_bull_regime': score.in_bull_regime
                })
                
            except Exception as e:
                print(f"    Error analyzing {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(results)
        df = df.sort_values('overall_score', ascending=False)
        
        self.results = df
        
        return df
    
    def get_recommendations(self, min_score: float = 60) -> pd.DataFrame:
        """
        Get stocks with recommendation of GOOD or EXCELLENT.
        
        Args:
            min_score: Minimum overall score
            
        Returns:
            Filtered DataFrame
        """
        if self.results.empty:
            return pd.DataFrame()
        
        return self.results[self.results['overall_score'] >= min_score]


def run_prerequisite_check(ticker: str, 
                           start_date: str = "2021-08-25",
                           verbose: bool = True) -> Tuple[StockSuitabilityScore, bool]:
    """
    Run prerequisite check for a single stock.
    
    Args:
        ticker: Stock ticker
        start_date: Start date for analysis
        verbose: Print detailed output
        
    Returns:
        Tuple of (StockSuitabilityScore, should_proceed)
    """
    import yfinance as yf
    
    print(f"\n{'='*60}")
    print(f"PREREQUISITE CHECK: {ticker}")
    print(f"{'='*60}")
    
    # Fetch data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date)
    
    if len(df) < 100:
        print(f"ERROR: Insufficient data ({len(df)} days)")
        return None, False
    
    df.columns = [c.lower() for c in df.columns]
    df.name = ticker
    
    # Get market data
    market = yf.Ticker("^GSPC")
    market_df = market.history(start=start_date)
    market_returns = market_df['Close'].pct_change().dropna()
    
    # Run analysis
    screener = StockSuitabilityScreener(df, market_df)
    score = screener.analyze(market_returns)
    
    if verbose:
        print(score)
    
    # Decision
    should_proceed = score.overall_score >= 50
    
    print(f"\n{'='*60}")
    if should_proceed:
        print(f"✓ PROCEED WITH ANALYSIS - Score: {score.overall_score:.1f}")
        print(f"  Stock meets prerequisites for anomaly prediction system")
    else:
        print(f"✗ SKIP ANALYSIS - Score: {score.overall_score:.1f}")
        print(f"  Stock does not meet prerequisites")
        print(f"  Consider: Higher beta, more trending stocks")
    print(f"{'='*60}")
    
    return score, should_proceed


if __name__ == "__main__":
    # Test on our three stocks
    tickers = ['RKLB', 'NVDA', 'BABA']
    
    for ticker in tickers:
        score, proceed = run_prerequisite_check(ticker)
