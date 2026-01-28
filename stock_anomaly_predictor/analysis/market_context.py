"""
Market Context Analysis Module

Analyzes broader market conditions and correlations:
- VIX (Volatility Index) analysis
- Market regime detection
- Sector rotation analysis
- Cross-asset correlations
- Risk-on/Risk-off indicators
- Market breadth indicators
- Correlation breakdown detection
- Beta and systematic risk analysis
- Liquidity indicators

These factors help contextualize individual stock signals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class MarketContextAnalyzer:
    """
    Analyze market context and correlations for signal enhancement.
    """
    
    def __init__(self, 
                 primary_df: pd.DataFrame,
                 market_data: Dict[str, pd.DataFrame],
                 related_data: Dict[str, pd.DataFrame] = None):
        """
        Initialize with market data.
        
        Args:
            primary_df: Primary stock DataFrame
            market_data: Dict of market indicator DataFrames
            related_data: Dict of related stock DataFrames
        """
        self.primary_df = primary_df.copy()
        self.market_data = market_data
        self.related_data = related_data or {}
        self.signals = pd.DataFrame(index=primary_df.index)
        
    def analyze_vix(self) -> pd.DataFrame:
        """
        Analyze VIX for volatility regime and contrarian signals.
        
        VIX Analysis:
        - High VIX (>30): Fear, potential capitulation
        - Low VIX (<15): Complacency, potential top
        - VIX spikes often precede reversals
        - VIX term structure (contango/backwardation)
        
        Returns:
            DataFrame with VIX-based signals
        """
        print("Analyzing VIX...")
        
        if '^VIX' not in self.market_data:
            print("  VIX data not available")
            return pd.DataFrame()
        
        vix_df = self.market_data['^VIX']
        
        if 'close' not in vix_df.columns:
            return pd.DataFrame()
        
        vix = vix_df['close']
        
        # Align with primary
        vix_aligned = vix.reindex(self.primary_df.index, method='ffill')
        
        # VIX levels
        self.signals['vix'] = vix_aligned
        
        # VIX moving averages
        self.signals['vix_sma_10'] = vix_aligned.rolling(10).mean()
        self.signals['vix_sma_20'] = vix_aligned.rolling(20).mean()
        self.signals['vix_sma_50'] = vix_aligned.rolling(50).mean()
        
        # VIX percentile (historical context)
        self.signals['vix_percentile'] = vix_aligned.rolling(252).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 20 else 0.5
        )
        
        # VIX regime
        self.signals['vix_regime'] = pd.cut(
            self.signals['vix_percentile'],
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['low_vol', 'normal_low', 'normal_high', 'high_vol']
        )
        
        # VIX spike detection
        vix_change = vix_aligned.pct_change()
        self.signals['vix_spike'] = vix_change > vix_change.rolling(50).std() * 2
        
        # VIX mean reversion signal
        # High VIX often mean reverts (bullish for stocks)
        vix_zscore = (vix_aligned - vix_aligned.rolling(50).mean()) / vix_aligned.rolling(50).std()
        self.signals['vix_zscore'] = vix_zscore
        
        # Contrarian VIX signal
        # High VIX = bullish contrarian, Low VIX = bearish contrarian
        self.signals['vix_contrarian_signal'] = -vix_zscore.clip(-2, 2) / 2
        
        # VIX crossover signals
        self.signals['vix_above_20'] = vix_aligned > 20
        self.signals['vix_above_30'] = vix_aligned > 30
        self.signals['vix_below_15'] = vix_aligned < 15
        
        print(f"  Current VIX: {vix_aligned.iloc[-1]:.1f}")
        print(f"  VIX percentile: {self.signals['vix_percentile'].iloc[-1]:.2%}")
        
        return self.signals[[c for c in self.signals.columns if 'vix' in c.lower()]]
    
    def analyze_market_regime(self) -> pd.DataFrame:
        """
        Determine overall market regime.
        
        Regimes:
        - Risk-On: Low VIX, positive breadth, trending up
        - Risk-Off: High VIX, negative breadth, trending down
        - Transitioning: Mixed signals
        
        Returns:
            DataFrame with market regime signals
        """
        print("Analyzing market regime...")
        
        # S&P 500 analysis
        if '^GSPC' in self.market_data:
            sp500 = self.market_data['^GSPC']
            sp500_close = sp500['close'].reindex(self.primary_df.index, method='ffill')
            
            # Trend
            sp500_sma_50 = sp500_close.rolling(50).mean()
            sp500_sma_200 = sp500_close.rolling(200).mean()
            
            self.signals['sp500_above_50sma'] = sp500_close > sp500_sma_50
            self.signals['sp500_above_200sma'] = sp500_close > sp500_sma_200
            self.signals['sp500_golden_cross'] = sp500_sma_50 > sp500_sma_200
            
            # S&P momentum
            self.signals['sp500_momentum_20'] = sp500_close.pct_change(20)
            
        # NASDAQ analysis
        if '^IXIC' in self.market_data:
            nasdaq = self.market_data['^IXIC']
            nasdaq_close = nasdaq['close'].reindex(self.primary_df.index, method='ffill')
            
            self.signals['nasdaq_momentum_20'] = nasdaq_close.pct_change(20)
            
            # Tech strength relative to broad market
            if '^GSPC' in self.market_data:
                sp500_close = self.market_data['^GSPC']['close'].reindex(self.primary_df.index, method='ffill')
                self.signals['tech_relative_strength'] = (
                    nasdaq_close.pct_change(20) - sp500_close.pct_change(20)
                )
        
        # Small cap analysis (Russell 2000)
        if 'IWM' in self.market_data:
            iwm = self.market_data['IWM']
            iwm_close = iwm['close'].reindex(self.primary_df.index, method='ffill')
            
            if '^GSPC' in self.market_data:
                sp500_close = self.market_data['^GSPC']['close'].reindex(self.primary_df.index, method='ffill')
                # Small cap relative strength (risk appetite indicator)
                self.signals['small_cap_relative'] = (
                    iwm_close.pct_change(20) - sp500_close.pct_change(20)
                )
        
        # Calculate overall risk-on/risk-off score
        risk_score = pd.Series(0.0, index=self.primary_df.index)
        
        if 'sp500_above_200sma' in self.signals.columns:
            risk_score += self.signals['sp500_above_200sma'].astype(float) * 0.3
        
        if 'sp500_momentum_20' in self.signals.columns:
            risk_score += (self.signals['sp500_momentum_20'] > 0).astype(float) * 0.2
        
        if 'vix_contrarian_signal' in self.signals.columns:
            risk_score += (self.signals['vix_contrarian_signal'] > 0).astype(float) * 0.2
        
        if 'tech_relative_strength' in self.signals.columns:
            risk_score += (self.signals['tech_relative_strength'] > 0).astype(float) * 0.15
        
        if 'small_cap_relative' in self.signals.columns:
            risk_score += (self.signals['small_cap_relative'] > 0).astype(float) * 0.15
        
        self.signals['risk_on_score'] = risk_score
        
        # Regime classification
        regime = pd.cut(
            risk_score,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['risk_off', 'neutral', 'risk_on']
        )
        self.signals['market_regime'] = regime
        
        print(f"  Current market regime: {regime.iloc[-1]}")
        print(f"  Risk-on score: {risk_score.iloc[-1]:.2f}")
        
        return self.signals
    
    def analyze_sector_rotation(self) -> pd.DataFrame:
        """
        Analyze sector rotation and space sector performance.
        
        Returns:
            DataFrame with sector analysis signals
        """
        print("Analyzing sector rotation...")
        
        # Space/Aerospace sector (ARKX)
        if 'ARKX' in self.market_data:
            arkx = self.market_data['ARKX']
            arkx_close = arkx['close'].reindex(self.primary_df.index, method='ffill')
            arkx_returns = arkx_close.pct_change()
            
            self.signals['space_sector_momentum'] = arkx_close.pct_change(20)
            
            # Correlation with sector
            primary_returns = self.primary_df['returns'] if 'returns' in self.primary_df.columns else self.primary_df['close'].pct_change()
            
            self.signals['sector_correlation'] = primary_returns.rolling(30).corr(arkx_returns)
            
            # Relative performance vs sector
            self.signals['vs_sector_performance'] = (
                self.primary_df['close'].pct_change(20) - arkx_close.pct_change(20)
            )
        
        # Energy sector (for launch cost correlation)
        if 'XLE' in self.market_data:
            xle = self.market_data['XLE']
            xle_close = xle['close'].reindex(self.primary_df.index, method='ffill')
            
            self.signals['energy_sector_momentum'] = xle_close.pct_change(20)
        
        # Oil (USO) - affects launch economics
        if 'USO' in self.market_data:
            uso = self.market_data['USO']
            uso_close = uso['close'].reindex(self.primary_df.index, method='ffill')
            
            self.signals['oil_momentum'] = uso_close.pct_change(20)
            self.signals['oil_level'] = uso_close
            
            # High oil is generally negative for aerospace
            oil_zscore = (uso_close - uso_close.rolling(60).mean()) / uso_close.rolling(60).std()
            self.signals['oil_zscore'] = oil_zscore
        
        return self.signals
    
    def analyze_peer_correlation(self) -> pd.DataFrame:
        """
        Analyze correlation with peer companies.
        
        Returns:
            DataFrame with peer correlation signals
        """
        print("Analyzing peer correlations...")
        
        if not self.related_data:
            print("  No peer data available")
            return self.signals
        
        primary_returns = (
            self.primary_df['returns'] 
            if 'returns' in self.primary_df.columns 
            else self.primary_df['close'].pct_change()
        )
        
        peer_correlations = {}
        peer_returns_list = []
        
        for ticker, df in self.related_data.items():
            if df.empty or 'close' not in df.columns:
                continue
            
            peer_returns = df['close'].pct_change().reindex(self.primary_df.index, method='ffill')
            peer_returns_list.append(peer_returns)
            
            # Rolling correlation
            corr = primary_returns.rolling(30).corr(peer_returns)
            peer_correlations[ticker] = corr
            
            self.signals[f'corr_{ticker}'] = corr
        
        if peer_correlations:
            # Average peer correlation
            corr_df = pd.DataFrame(peer_correlations)
            self.signals['avg_peer_correlation'] = corr_df.mean(axis=1)
            
            # Correlation breakdown (decorrelation from peers might signal idiosyncratic move)
            historical_corr = self.signals['avg_peer_correlation'].rolling(60).mean()
            corr_deviation = self.signals['avg_peer_correlation'] - historical_corr
            self.signals['correlation_breakdown'] = corr_deviation < -0.2
        
        # Relative strength vs peers
        if peer_returns_list:
            peer_avg_returns = pd.concat(peer_returns_list, axis=1).mean(axis=1)
            self.signals['vs_peers_performance'] = primary_returns.rolling(20).sum() - peer_avg_returns.rolling(20).sum()
            
            # Is stock leading or lagging peers?
            self.signals['leading_peers'] = self.signals['vs_peers_performance'] > 0.05
        
        return self.signals
    
    def calculate_beta_analysis(self) -> pd.DataFrame:
        """
        Calculate rolling beta and systematic risk metrics.
        
        Returns:
            DataFrame with beta analysis
        """
        print("Calculating beta analysis...")
        
        if '^GSPC' not in self.market_data:
            return self.signals
        
        sp500 = self.market_data['^GSPC']
        sp500_returns = sp500['close'].pct_change().reindex(self.primary_df.index, method='ffill')
        
        primary_returns = (
            self.primary_df['returns'] 
            if 'returns' in self.primary_df.columns 
            else self.primary_df['close'].pct_change()
        )
        
        # Rolling beta calculation
        def calc_beta(window_data):
            if len(window_data) < 20:
                return np.nan
            
            stock = window_data.iloc[:, 0]
            market = window_data.iloc[:, 1]
            
            cov = stock.cov(market)
            var = market.var()
            
            return cov / var if var > 0 else np.nan
        
        combined = pd.concat([primary_returns, sp500_returns], axis=1)
        combined.columns = ['stock', 'market']
        
        # 60-day rolling beta
        betas = []
        for i in range(60, len(combined)):
            window = combined.iloc[i-60:i]
            beta = calc_beta(window)
            betas.append(beta)
        
        betas = [np.nan] * 60 + betas
        self.signals['beta_60'] = pd.Series(betas, index=self.primary_df.index)
        
        # Beta regime
        beta = self.signals['beta_60']
        self.signals['high_beta'] = beta > 1.5
        self.signals['low_beta'] = beta < 0.8
        
        # Beta change (increasing beta might signal momentum)
        self.signals['beta_change'] = beta.diff(20)
        
        # Alpha (excess return over beta-adjusted market return)
        expected_return = beta * sp500_returns
        self.signals['rolling_alpha'] = (primary_returns - expected_return).rolling(20).sum()
        
        print(f"  Current beta: {beta.iloc[-1]:.2f}")
        
        return self.signals
    
    def calculate_composite_market_signal(self) -> pd.Series:
        """
        Calculate composite market context signal.
        
        Returns:
            Series of composite market signal (-1 to 1)
        """
        print("Calculating composite market signal...")
        
        signal = pd.Series(0.0, index=self.primary_df.index)
        weights_used = 0
        
        # VIX component (contrarian)
        if 'vix_contrarian_signal' in self.signals.columns:
            signal += self.signals['vix_contrarian_signal'].fillna(0) * 0.25
            weights_used += 0.25
        
        # Market regime
        if 'risk_on_score' in self.signals.columns:
            # Normalize risk_on_score to -1 to 1
            ros = (self.signals['risk_on_score'] - 0.5) * 2
            signal += ros.fillna(0) * 0.25
            weights_used += 0.25
        
        # Sector momentum
        if 'space_sector_momentum' in self.signals.columns:
            sector_mom = self.signals['space_sector_momentum'].fillna(0).clip(-0.1, 0.1) * 5
            signal += sector_mom * 0.15
            weights_used += 0.15
        
        # Relative strength vs peers
        if 'vs_peers_performance' in self.signals.columns:
            rel_str = self.signals['vs_peers_performance'].fillna(0).clip(-0.2, 0.2) * 2.5
            signal += rel_str * 0.15
            weights_used += 0.15
        
        # Alpha component
        if 'rolling_alpha' in self.signals.columns:
            alpha = self.signals['rolling_alpha'].fillna(0).clip(-0.1, 0.1) * 5
            signal += alpha * 0.10
            weights_used += 0.10
        
        # Correlation breakdown (idiosyncratic opportunity)
        if 'correlation_breakdown' in self.signals.columns:
            decorr = self.signals['correlation_breakdown'].fillna(False).astype(float)
            signal += decorr * 0.10
            weights_used += 0.10
        
        # Normalize if not all components available
        if weights_used > 0:
            signal = signal / weights_used
        
        self.signals['market_context_signal'] = signal.clip(-1, 1)
        
        return self.signals['market_context_signal']
    
    def run_full_analysis(self) -> pd.DataFrame:
        """
        Run complete market context analysis.
        
        Returns:
            DataFrame with all market context signals
        """
        print("=" * 60)
        print("Running Market Context Analysis")
        print("=" * 60)
        
        # VIX analysis
        self.analyze_vix()
        
        # Market regime
        self.analyze_market_regime()
        
        # Sector rotation
        self.analyze_sector_rotation()
        
        # Peer correlations
        self.analyze_peer_correlation()
        
        # Beta analysis
        self.calculate_beta_analysis()
        
        # Composite signal
        self.calculate_composite_market_signal()
        
        print("=" * 60)
        print("Market Context Analysis Complete")
        print(f"Generated {len(self.signals.columns)} market context signals")
        print("=" * 60)
        
        return self.signals
    
    def get_market_summary(self) -> Dict:
        """Get current market context summary."""
        if self.signals.empty:
            return {}
        
        latest = self.signals.iloc[-1]
        
        summary = {
            'vix': latest.get('vix', np.nan),
            'vix_percentile': latest.get('vix_percentile', np.nan),
            'market_regime': latest.get('market_regime', 'unknown'),
            'risk_on_score': latest.get('risk_on_score', np.nan),
            'beta': latest.get('beta_60', np.nan),
            'sector_momentum': latest.get('space_sector_momentum', np.nan),
            'vs_peers': latest.get('vs_peers_performance', np.nan),
            'market_context_signal': latest.get('market_context_signal', np.nan)
        }
        
        return summary


if __name__ == "__main__":
    # Test market context analysis
    from data.collector import DataCollector
    
    # Collect all data
    collector = DataCollector("RKLB")
    data = collector.collect_all_data()
    
    # Run market analysis
    analyzer = MarketContextAnalyzer(
        data['primary'],
        data['market'],
        data['related']
    )
    
    signals = analyzer.run_full_analysis()
    summary = analyzer.get_market_summary()
    
    print("\nMarket Context Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
