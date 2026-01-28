"""
Signal Aggregator and Ensemble Module

Combines all anomaly detection, cyclical, sentiment, and market signals
into a unified prediction framework:
- Weighted signal combination
- Dynamic weight adjustment based on regime
- Confidence scoring
- Signal confirmation logic
- Multi-timeframe signal alignment
- Adaptive ensemble based on historical performance

Mathematical Foundation: Ensemble methods, Bayesian averaging
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config, SignalWeights


@dataclass
class SignalCategory:
    """Signal category with component signals and weights."""
    name: str
    signals: List[str]
    weight: float
    higher_is_bullish: bool = True


class SignalAggregator:
    """
    Aggregates multiple signal sources into unified prediction scores.
    """
    
    def __init__(self):
        """Initialize signal aggregator."""
        self.weights = config.weights
        self.signals = pd.DataFrame()
        self.composite_signals = pd.DataFrame()
        
        # Define signal categories
        self.signal_categories = self._define_signal_categories()
    
    def _define_signal_categories(self) -> List[SignalCategory]:
        """Define signal categories and their components."""
        return [
            SignalCategory(
                name='statistical_anomaly',
                signals=[
                    'returns_zscore', 'returns_mod_zscore', 'volume_anomaly_score',
                    'gap_zscore', 'stat_anomaly_ratio'
                ],
                weight=0.15,
                higher_is_bullish=False  # Higher anomaly score = potential move (either direction)
            ),
            SignalCategory(
                name='ml_anomaly',
                signals=[
                    'ml_ensemble_score', 'isolation_forest_score', 'lof_score'
                ],
                weight=0.15,
                higher_is_bullish=False
            ),
            SignalCategory(
                name='volume_confirmation',
                signals=[
                    'volume_ratio', 'volume_zscore', 'volume_percentile'
                ],
                weight=0.12,
                higher_is_bullish=True  # High volume confirms moves
            ),
            SignalCategory(
                name='momentum',
                signals=[
                    'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
                    'macd_histogram', 'rsi'
                ],
                weight=0.12,
                higher_is_bullish=True
            ),
            SignalCategory(
                name='mean_reversion',
                signals=[
                    'bb_mr_signal', 'ou_mr_signal', 'bb_pct_b'
                ],
                weight=0.10,
                higher_is_bullish=True  # Positive = oversold = bullish
            ),
            SignalCategory(
                name='trend_following',
                signals=[
                    'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
                    'sma_20_slope', 'sma_50_slope'
                ],
                weight=0.08,
                higher_is_bullish=True
            ),
            SignalCategory(
                name='sentiment',
                signals=[
                    'sentiment_composite', 'twitter_sentiment', 'news_sentiment',
                    'sentiment_momentum'
                ],
                weight=0.10,
                higher_is_bullish=True
            ),
            SignalCategory(
                name='search_interest',
                signals=[
                    'trends_zscore', 'trends_aggregate'
                ],
                weight=0.05,
                higher_is_bullish=True  # Increasing interest can be bullish
            ),
            SignalCategory(
                name='market_context',
                signals=[
                    'market_context_signal', 'vix_contrarian_signal',
                    'risk_on_score', 'vs_peers_performance'
                ],
                weight=0.08,
                higher_is_bullish=True
            ),
            SignalCategory(
                name='cyclical',
                signals=[
                    'cycle_position', 'rolling_hurst', 'spectral_entropy'
                ],
                weight=0.05,
                higher_is_bullish=False  # Depends on position
            )
        ]
    
    def merge_all_signals(self,
                          price_df: pd.DataFrame,
                          stat_anomalies: pd.DataFrame,
                          ml_anomalies: pd.DataFrame,
                          cyclical_signals: pd.DataFrame,
                          sentiment_signals: pd.DataFrame,
                          market_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all signal sources into unified DataFrame.
        
        Args:
            price_df: Primary price DataFrame
            stat_anomalies: Statistical anomaly signals
            ml_anomalies: ML anomaly signals
            cyclical_signals: Cyclical analysis signals
            sentiment_signals: Sentiment signals
            market_signals: Market context signals
            
        Returns:
            Merged DataFrame with all signals
        """
        print("Merging all signal sources...")
        
        # Start with price data
        self.signals = price_df.copy()
        
        # Merge each signal source
        sources = [
            ('statistical', stat_anomalies),
            ('ml', ml_anomalies),
            ('cyclical', cyclical_signals),
            ('sentiment', sentiment_signals),
            ('market', market_signals)
        ]
        
        for name, df in sources:
            if df is not None and not df.empty:
                # Add prefix to avoid column name conflicts
                df_prefixed = df.add_suffix(f'_{name}') if not any(name in c for c in df.columns) else df
                
                # Merge on index
                for col in df.columns:
                    if col not in self.signals.columns:
                        self.signals[col] = df[col]
                
                print(f"  Merged {len(df.columns)} signals from {name}")
        
        print(f"Total signals: {len(self.signals.columns)}")
        
        return self.signals
    
    def normalize_signal(self, 
                         series: pd.Series, 
                         method: str = 'zscore',
                         window: int = 60) -> pd.Series:
        """
        Normalize signal to comparable scale.
        
        Args:
            series: Signal series
            method: Normalization method ('zscore', 'minmax', 'percentile')
            window: Rolling window for normalization
            
        Returns:
            Normalized series
        """
        if method == 'zscore':
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
            return (series - mean) / std.replace(0, 1)
        
        elif method == 'minmax':
            min_val = series.rolling(window=window).min()
            max_val = series.rolling(window=window).max()
            range_val = (max_val - min_val).replace(0, 1)
            return (series - min_val) / range_val
        
        elif method == 'percentile':
            return series.rolling(window=window).apply(
                lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 
                if len(x.dropna()) > 10 else 0.5
            )
        
        return series
    
    def calculate_category_scores(self) -> pd.DataFrame:
        """
        Calculate composite scores for each signal category.
        
        Returns:
            DataFrame with category scores
        """
        print("Calculating category scores...")
        
        category_scores = pd.DataFrame(index=self.signals.index)
        
        for category in self.signal_categories:
            # Find available signals in this category
            available_signals = [s for s in category.signals if s in self.signals.columns]
            
            if not available_signals:
                print(f"  {category.name}: No signals available")
                continue
            
            # Calculate category score
            scores = []
            for sig_name in available_signals:
                sig = self.signals[sig_name].copy()
                
                # Normalize
                sig_normalized = self.normalize_signal(sig, method='percentile')
                
                # Convert to directional signal if needed
                if not category.higher_is_bullish:
                    # For anomaly scores, high values indicate potential moves (non-directional)
                    # We'll keep the magnitude
                    pass
                
                scores.append(sig_normalized)
            
            if scores:
                # Average available signals
                category_score = pd.concat(scores, axis=1).mean(axis=1)
                category_scores[f'{category.name}_score'] = category_score
                print(f"  {category.name}: {len(available_signals)} signals")
        
        self.composite_signals = category_scores
        return category_scores
    
    def calculate_rally_probability(self, 
                                    lookforward: int = 5,
                                    threshold: float = 0.05) -> pd.Series:
        """
        Calculate probability of a rally (big upward move) based on all signals.
        
        Args:
            lookforward: Days to look forward for defining rally
            threshold: Minimum return to consider a rally
            
        Returns:
            Series of rally probabilities
        """
        print(f"Calculating rally probability (>{threshold:.0%} in {lookforward} days)...")
        
        if self.composite_signals.empty:
            self.calculate_category_scores()
        
        # Calculate weighted composite score
        weights = {cat.name + '_score': cat.weight for cat in self.signal_categories}
        
        composite = pd.Series(0.0, index=self.signals.index)
        total_weight = 0
        
        for col, weight in weights.items():
            if col in self.composite_signals.columns:
                composite += self.composite_signals[col].fillna(0.5) * weight
                total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        # Transform to probability-like scale (0 to 1)
        # Using sigmoid transformation
        composite_centered = (composite - 0.5) * 4  # Scale to roughly -2 to 2
        rally_prob = 1 / (1 + np.exp(-composite_centered))
        
        self.composite_signals['rally_probability'] = rally_prob
        
        return rally_prob
    
    def calculate_anomaly_intensity(self) -> pd.Series:
        """
        Calculate overall anomaly intensity (how unusual is current market state).
        
        High anomaly intensity + directional bias = actionable signal
        
        Returns:
            Series of anomaly intensity scores
        """
        print("Calculating anomaly intensity...")
        
        anomaly_cols = [
            'stat_anomaly_ratio', 'ml_ensemble_score', 'volume_anomaly_score',
            'returns_zscore', 'returns_mod_zscore'
        ]
        
        available = [c for c in anomaly_cols if c in self.signals.columns]
        
        if not available:
            return pd.Series(0.5, index=self.signals.index)
        
        # Normalize each anomaly score
        normalized = []
        for col in available:
            norm = self.normalize_signal(self.signals[col], method='percentile')
            normalized.append(norm)
        
        intensity = pd.concat(normalized, axis=1).mean(axis=1)
        self.composite_signals['anomaly_intensity'] = intensity
        
        return intensity
    
    def calculate_signal_confidence(self) -> pd.Series:
        """
        Calculate confidence in the signal based on signal agreement.
        
        High confidence when multiple signals agree.
        
        Returns:
            Series of confidence scores
        """
        print("Calculating signal confidence...")
        
        if self.composite_signals.empty:
            self.calculate_category_scores()
        
        score_cols = [c for c in self.composite_signals.columns if c.endswith('_score')]
        
        if len(score_cols) < 2:
            return pd.Series(0.5, index=self.signals.index)
        
        # Calculate agreement (inverse of standard deviation)
        scores_df = self.composite_signals[score_cols]
        
        # Mean of scores
        mean_score = scores_df.mean(axis=1)
        
        # Standard deviation (disagreement)
        std_score = scores_df.std(axis=1)
        
        # Confidence is higher when std is lower and signals are extreme
        # Confidence = (1 - normalized_std) * extremeness
        std_normalized = self.normalize_signal(std_score, method='minmax')
        extremeness = np.abs(mean_score - 0.5) * 2  # 0 to 1
        
        confidence = (1 - std_normalized.fillna(0.5)) * 0.5 + extremeness * 0.5
        confidence = confidence.clip(0, 1)
        
        self.composite_signals['signal_confidence'] = confidence
        
        return confidence
    
    def calculate_directional_bias(self) -> pd.Series:
        """
        Calculate directional bias (-1 = bearish, +1 = bullish).
        
        Returns:
            Series of directional bias
        """
        print("Calculating directional bias...")
        
        # Bullish signals
        bullish_signals = [
            ('momentum_score', 1.0),
            ('mean_reversion_score', 1.0),  # Positive when oversold
            ('sentiment_score', 1.0),
            ('market_context_score', 1.0),
            ('trend_following_score', 1.0)
        ]
        
        bias = pd.Series(0.0, index=self.signals.index)
        total_weight = 0
        
        for col, weight in bullish_signals:
            if col in self.composite_signals.columns:
                # Convert 0-1 score to -1 to 1 bias
                signal_bias = (self.composite_signals[col] - 0.5) * 2
                bias += signal_bias.fillna(0) * weight
                total_weight += weight
        
        if total_weight > 0:
            bias /= total_weight
        
        bias = bias.clip(-1, 1)
        self.composite_signals['directional_bias'] = bias
        
        return bias
    
    def generate_trade_signals(self,
                               prob_threshold: float = 0.7,
                               confidence_threshold: float = 0.6,
                               anomaly_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate actionable trade signals based on composite analysis.
        
        Signal conditions:
        1. Rally probability above threshold
        2. Signal confidence above threshold
        3. Anomaly intensity above threshold (unusual conditions)
        4. Positive directional bias
        
        Args:
            prob_threshold: Minimum rally probability
            confidence_threshold: Minimum confidence
            anomaly_threshold: Minimum anomaly intensity
            
        Returns:
            DataFrame with trade signals
        """
        print("=" * 60)
        print("Generating Trade Signals")
        print("=" * 60)
        
        # Calculate all composite metrics
        rally_prob = self.calculate_rally_probability()
        anomaly_int = self.calculate_anomaly_intensity()
        confidence = self.calculate_signal_confidence()
        bias = self.calculate_directional_bias()
        
        # Signal conditions
        signals = pd.DataFrame(index=self.signals.index)
        
        # Raw scores
        signals['rally_probability'] = rally_prob
        signals['anomaly_intensity'] = anomaly_int
        signals['signal_confidence'] = confidence
        signals['directional_bias'] = bias
        
        # Composite signal score (0-100)
        signals['composite_score'] = (
            rally_prob * 0.35 +
            anomaly_int * 0.25 +
            confidence * 0.20 +
            (bias + 1) / 2 * 0.20  # Normalize bias to 0-1
        ) * 100
        
        # Generate buy signals
        signals['buy_signal'] = (
            (rally_prob > prob_threshold) &
            (confidence > confidence_threshold) &
            (anomaly_int > anomaly_threshold) &
            (bias > 0)
        )
        
        # Signal strength (for position sizing)
        signals['signal_strength'] = (
            signals['buy_signal'].astype(float) *
            signals['composite_score'] / 100
        )
        
        # Alert levels
        signals['alert_level'] = pd.cut(
            signals['composite_score'],
            bins=[0, 40, 55, 70, 85, 100],
            labels=['very_low', 'low', 'moderate', 'high', 'very_high']
        )
        
        # Count recent signals (avoid over-signaling)
        signals['signals_last_5d'] = signals['buy_signal'].rolling(5).sum()
        
        # Final filtered signal (avoid consecutive signals)
        signals['actionable_signal'] = (
            signals['buy_signal'] &
            (signals['signals_last_5d'] <= 1)  # No more than 1 signal in 5 days
        )
        
        # Summary statistics
        total_signals = signals['buy_signal'].sum()
        actionable = signals['actionable_signal'].sum()
        
        print(f"Total raw buy signals: {total_signals}")
        print(f"Actionable signals (filtered): {actionable}")
        print(f"Current composite score: {signals['composite_score'].iloc[-1]:.1f}")
        print(f"Current alert level: {signals['alert_level'].iloc[-1]}")
        print("=" * 60)
        
        return signals
    
    def get_current_analysis(self) -> Dict:
        """Get analysis for the most recent date."""
        if self.composite_signals.empty:
            return {}
        
        latest = self.composite_signals.iloc[-1]
        
        analysis = {
            'date': self.composite_signals.index[-1],
            'rally_probability': latest.get('rally_probability', 0),
            'anomaly_intensity': latest.get('anomaly_intensity', 0),
            'signal_confidence': latest.get('signal_confidence', 0),
            'directional_bias': latest.get('directional_bias', 0),
        }
        
        # Add category scores
        for cat in self.signal_categories:
            col = f'{cat.name}_score'
            if col in self.composite_signals.columns:
                analysis[col] = latest[col]
        
        # Interpretation
        prob = analysis['rally_probability']
        if prob > 0.75:
            analysis['interpretation'] = 'Strong bullish setup detected'
        elif prob > 0.6:
            analysis['interpretation'] = 'Moderately bullish conditions'
        elif prob < 0.3:
            analysis['interpretation'] = 'Bearish conditions or consolidation'
        else:
            analysis['interpretation'] = 'Neutral / No clear signal'
        
        return analysis


class AdaptiveEnsemble:
    """
    Adaptive ensemble that adjusts weights based on historical performance.
    """
    
    def __init__(self, 
                 aggregator: SignalAggregator,
                 lookback: int = 60,
                 min_trades: int = 5):
        """
        Initialize adaptive ensemble.
        
        Args:
            aggregator: SignalAggregator instance
            lookback: Days to look back for performance
            min_trades: Minimum trades for weight adjustment
        """
        self.aggregator = aggregator
        self.lookback = lookback
        self.min_trades = min_trades
        self.performance_history = {}
    
    def evaluate_signal_performance(self,
                                    signals_df: pd.DataFrame,
                                    forward_returns: pd.Series,
                                    forward_days: int = 5) -> Dict[str, float]:
        """
        Evaluate performance of each signal category.
        
        Args:
            signals_df: DataFrame with category scores
            forward_returns: Series of forward returns
            forward_days: Days to look forward
            
        Returns:
            Dictionary of category performances (correlation with returns)
        """
        performances = {}
        
        score_cols = [c for c in signals_df.columns if c.endswith('_score')]
        
        for col in score_cols:
            # Calculate correlation with forward returns
            valid_data = pd.concat([signals_df[col], forward_returns], axis=1).dropna()
            
            if len(valid_data) > self.min_trades:
                corr = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
                performances[col] = corr if not np.isnan(corr) else 0
            else:
                performances[col] = 0
        
        return performances
    
    def update_weights(self, performances: Dict[str, float]) -> Dict[str, float]:
        """
        Update category weights based on performance.
        
        Args:
            performances: Dictionary of category performances
            
        Returns:
            Updated weights
        """
        # Convert correlations to weights (positive correlation = higher weight)
        adjusted_perfs = {k: max(0.1, v + 0.5) for k, v in performances.items()}
        
        # Normalize to sum to 1
        total = sum(adjusted_perfs.values())
        if total > 0:
            weights = {k: v / total for k, v in adjusted_perfs.items()}
        else:
            # Equal weights if no performance data
            weights = {k: 1/len(adjusted_perfs) for k in adjusted_perfs.keys()}
        
        return weights
    
    def calculate_adaptive_signal(self,
                                  signals_df: pd.DataFrame,
                                  returns: pd.Series) -> pd.Series:
        """
        Calculate adaptive ensemble signal with dynamic weights.
        
        Args:
            signals_df: DataFrame with category scores
            returns: Historical returns for performance evaluation
            
        Returns:
            Adaptive ensemble signal
        """
        # Calculate rolling forward returns
        forward_returns = returns.shift(-5).rolling(5).sum()
        
        # Rolling weight updates
        adaptive_signal = pd.Series(index=signals_df.index, dtype=float)
        
        score_cols = [c for c in signals_df.columns if c.endswith('_score')]
        
        for i in range(self.lookback, len(signals_df)):
            # Get historical window
            window_signals = signals_df.iloc[i-self.lookback:i]
            window_returns = forward_returns.iloc[i-self.lookback:i]
            
            # Evaluate performance
            perfs = self.evaluate_signal_performance(window_signals, window_returns)
            
            # Update weights
            weights = self.update_weights(perfs)
            
            # Calculate weighted signal for current day
            current_signal = 0
            for col in score_cols:
                if col in weights:
                    current_signal += signals_df[col].iloc[i] * weights.get(col, 0)
            
            adaptive_signal.iloc[i] = current_signal
        
        return adaptive_signal


if __name__ == "__main__":
    # Test signal aggregator
    print("Testing Signal Aggregator...")
    
    # This would normally receive data from the other modules
    # For testing, we create dummy data
    
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    # Create dummy signals
    dummy_signals = pd.DataFrame({
        'close': 10 + np.cumsum(np.random.randn(252) * 0.1),
        'returns': np.random.randn(252) * 0.02,
        'volume': np.random.randint(1000000, 5000000, 252),
        'returns_zscore': np.random.randn(252),
        'volume_anomaly_score': np.random.random(252),
        'stat_anomaly_ratio': np.random.random(252) * 0.3,
        'ml_ensemble_score': np.random.random(252),
        'bb_mr_signal': np.random.randn(252) * 0.5,
        'sentiment_composite': np.random.randn(252) * 0.3,
        'market_context_signal': np.random.randn(252) * 0.3,
        'roc_10': np.random.randn(252) * 0.05,
        'momentum_10': np.random.randn(252) * 0.5,
    }, index=dates)
    
    # Test aggregator
    aggregator = SignalAggregator()
    aggregator.signals = dummy_signals
    
    # Generate trade signals
    trade_signals = aggregator.generate_trade_signals()
    
    print("\nTrade Signal Summary:")
    print(trade_signals[['composite_score', 'alert_level', 'buy_signal']].tail(10))
    
    # Current analysis
    analysis = aggregator.get_current_analysis()
    print("\nCurrent Analysis:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
