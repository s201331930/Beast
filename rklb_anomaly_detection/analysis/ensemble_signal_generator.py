"""
Ensemble Signal Generator
=========================
Combines all anomaly detection models into a unified signal system:
- Statistical anomaly ensemble
- Machine learning ensemble
- Cyclical/mean reversion ensemble
- Sentiment ensemble
- Technical indicator confirmation
- Final weighted signal aggregation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import all model ensembles
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.statistical_models import StatisticalAnomalyEnsemble
from models.ml_models import MLAnomalyEnsemble
from models.cyclical_models import CyclicalModelEnsemble
from analysis.technical_indicators import TechnicalIndicators
from analysis.sentiment_analyzer import SentimentAnalyzer, VIXAnalyzer


@dataclass
class SignalConfig:
    """Configuration for ensemble signal generation."""
    # Minimum models required to agree for signal
    min_stat_models: int = 2
    min_ml_models: int = 2
    min_cyclical_models: int = 2
    min_sentiment_signals: int = 1
    
    # Weights for different model categories
    stat_weight: float = 0.25
    ml_weight: float = 0.25
    cyclical_weight: float = 0.25
    sentiment_weight: float = 0.15
    technical_weight: float = 0.10
    
    # Signal thresholds
    signal_threshold: float = 0.5
    strong_signal_threshold: float = 0.7
    
    # Confirmation requirements
    require_volume_confirmation: bool = True
    require_trend_confirmation: bool = False
    
    # Cooldown between signals (days)
    signal_cooldown: int = 3


class EnsembleSignalGenerator:
    """
    Master signal generator combining all anomaly detection models.
    Produces unified buy/sell signals with confidence scores.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        
        # Initialize all ensembles
        self.stat_ensemble = StatisticalAnomalyEnsemble()
        self.ml_ensemble = MLAnomalyEnsemble()
        self.cyclical_ensemble = CyclicalModelEnsemble()
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Results storage
        self.all_features = None
        self.signals = None
        
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate ensemble signals from all models.
        
        Args:
            data: DataFrame with OHLCV data and any additional features
            
        Returns:
            Tuple of (all_features DataFrame, signals Series)
        """
        logger.info("Starting ensemble signal generation...")
        
        # Ensure required columns exist
        df = self._prepare_data(data)
        
        # Generate features from all models
        logger.info("Running statistical anomaly detection...")
        stat_features = self.stat_ensemble.fit_transform(df)
        
        logger.info("Running machine learning anomaly detection...")
        ml_features = self.ml_ensemble.fit_transform(df)
        
        logger.info("Running cyclical/mean reversion analysis...")
        cyclical_features = self.cyclical_ensemble.fit_transform(df)
        
        logger.info("Calculating technical indicators...")
        tech_features = self.tech_indicators.calculate_all(df)
        
        logger.info("Running sentiment analysis...")
        sentiment_features = self.sentiment_analyzer.analyze_all_sentiment(df)
        
        # Combine all features
        all_features = df.copy()
        
        for features_df in [stat_features, ml_features, cyclical_features, 
                           tech_features, sentiment_features]:
            if features_df is not None and not features_df.empty:
                # Only add new columns
                new_cols = [c for c in features_df.columns if c not in all_features.columns]
                if new_cols:
                    all_features = all_features.join(features_df[new_cols], how='left')
        
        self.all_features = all_features
        
        # Generate ensemble signals
        signals = self._generate_ensemble_signals(all_features)
        
        self.signals = signals
        
        logger.info(f"Signal generation complete. Total features: {len(all_features.columns)}")
        
        return all_features, signals
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with required columns."""
        df = data.copy()
        
        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate required features if missing
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        if 'volatility_20d' not in df.columns:
            df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        if 'volume_ratio' not in df.columns and 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        if 'price_range' not in df.columns:
            df['price_range'] = (df['high'] - df['low']) / df['close']
        
        if 'gap' not in df.columns:
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # ATR
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        if 'atr_14' not in df.columns:
            df['atr_14'] = tr.rolling(14).mean()
        
        return df
    
    def _generate_ensemble_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate final ensemble signals by combining all model outputs.
        """
        n = len(features)
        
        # Initialize signal components
        signal_components = pd.DataFrame(index=features.index)
        
        # 1. Statistical Model Signal
        stat_score = self._get_stat_signal(features)
        signal_components['stat_signal'] = stat_score
        
        # 2. ML Model Signal
        ml_score = self._get_ml_signal(features)
        signal_components['ml_signal'] = ml_score
        
        # 3. Cyclical/Mean Reversion Signal
        cyclical_score = self._get_cyclical_signal(features)
        signal_components['cyclical_signal'] = cyclical_score
        
        # 4. Sentiment Signal
        sentiment_score = self._get_sentiment_signal(features)
        signal_components['sentiment_signal'] = sentiment_score
        
        # 5. Technical Confirmation
        tech_score = self._get_technical_signal(features)
        signal_components['tech_signal'] = tech_score
        
        # Calculate weighted ensemble score
        weights = np.array([
            self.config.stat_weight,
            self.config.ml_weight,
            self.config.cyclical_weight,
            self.config.sentiment_weight,
            self.config.technical_weight
        ])
        
        signal_matrix = signal_components.values
        ensemble_score = np.nansum(signal_matrix * weights, axis=1)
        
        # Normalize to [-1, 1]
        ensemble_score = np.clip(ensemble_score / weights.sum(), -1, 1)
        
        # Apply threshold for final signals
        final_signals = pd.Series(0, index=features.index)
        
        # Strong buy signal
        final_signals[ensemble_score >= self.config.strong_signal_threshold] = 2
        
        # Regular buy signal
        final_signals[(ensemble_score >= self.config.signal_threshold) & 
                     (ensemble_score < self.config.strong_signal_threshold)] = 1
        
        # Apply confirmations if required
        if self.config.require_volume_confirmation:
            volume_confirmed = features.get('Volume_spike', 0) == 1
            # Only reduce signals that lack volume confirmation
            final_signals = final_signals.where(
                (final_signals == 0) | volume_confirmed,
                final_signals - 1
            ).clip(0)
        
        # Apply cooldown
        final_signals = self._apply_cooldown(final_signals)
        
        # Store component scores for analysis
        features['ensemble_score'] = ensemble_score
        features['stat_signal'] = signal_components['stat_signal']
        features['ml_signal'] = signal_components['ml_signal']
        features['cyclical_signal'] = signal_components['cyclical_signal']
        features['sentiment_signal'] = signal_components['sentiment_signal']
        features['tech_signal'] = signal_components['tech_signal']
        
        return final_signals
    
    def _get_stat_signal(self, features: pd.DataFrame) -> pd.Series:
        """Extract statistical anomaly signal."""
        signal = pd.Series(0.0, index=features.index)
        
        # Count statistical anomalies
        stat_cols = [c for c in features.columns if 'stat_' in c.lower() or 
                    c in ['ZScore_anomaly', 'Bollinger_anomaly', 'GARCH_anomaly']]
        
        if 'stat_anomaly_count' in features.columns:
            # Normalize count to [0, 1]
            max_count = features['stat_anomaly_count'].max()
            if max_count > 0:
                signal = features['stat_anomaly_count'] / max_count
        elif 'stat_anomaly_mean_score' in features.columns:
            signal = features['stat_anomaly_mean_score'].clip(0, 1)
        
        # Directional adjustment
        if 'Bollinger_lower_breakout' in features.columns:
            # Oversold = potential rally
            signal = signal.where(
                features.get('Bollinger_lower_breakout', 0) == 0,
                signal * 1.2
            )
        
        return signal.fillna(0)
    
    def _get_ml_signal(self, features: pd.DataFrame) -> pd.Series:
        """Extract ML anomaly signal."""
        signal = pd.Series(0.0, index=features.index)
        
        if 'ml_anomaly_count' in features.columns:
            max_count = features['ml_anomaly_count'].max()
            if max_count > 0:
                signal = features['ml_anomaly_count'] / max_count
        elif 'ml_anomaly_mean_score' in features.columns:
            signal = features['ml_anomaly_mean_score'].clip(0, 1)
        
        return signal.fillna(0)
    
    def _get_cyclical_signal(self, features: pd.DataFrame) -> pd.Series:
        """Extract cyclical/mean reversion signal."""
        signal = pd.Series(0.0, index=features.index)
        
        components = []
        
        # Hurst exponent - mean reverting regime
        if 'Hurst_mean_reverting' in features.columns:
            hurst_signal = features['Hurst_mean_reverting'].fillna(0) * 0.5
            components.append(hurst_signal)
        
        # OU process - distance from mean
        if 'OU_distance_from_mean' in features.columns:
            ou_signal = np.abs(features['OU_distance_from_mean']).clip(0, 3) / 3
            # Bullish when price is significantly below mean
            ou_direction = (features['OU_distance_from_mean'] < -1.5).astype(float)
            ou_signal = ou_signal * ou_direction
            components.append(ou_signal)
        
        # HMM regime - bullish probability
        if 'HMM_bull_prob' in features.columns:
            hmm_signal = features['HMM_bull_prob'].fillna(0.5)
            components.append(hmm_signal)
        
        # Fourier - cycle position
        if 'Fourier_cycle_position' in features.columns:
            # Signal stronger at cycle troughs (position near 0 or 1)
            cycle_pos = features['Fourier_cycle_position'].fillna(0.5)
            cycle_signal = 1 - np.abs(cycle_pos - 0.5) * 2  # Stronger at extremes
            components.append(cycle_signal * 0.5)
        
        if components:
            signal = pd.concat(components, axis=1).mean(axis=1)
        
        return signal.fillna(0).clip(0, 1)
    
    def _get_sentiment_signal(self, features: pd.DataFrame) -> pd.Series:
        """Extract sentiment-based signal."""
        signal = pd.Series(0.0, index=features.index)
        
        components = []
        
        # Extreme fear (contrarian bullish)
        if 'extreme_fear' in features.columns:
            fear_signal = features['extreme_fear'].fillna(0) * 0.8
            components.append(fear_signal)
        
        # PCR extreme fear
        if 'pcr_extreme_fear' in features.columns:
            pcr_signal = features['pcr_extreme_fear'].fillna(0) * 0.7
            components.append(pcr_signal)
        
        # VIX reversion
        if 'vix_reversion_signal' in features.columns:
            vix_signal = features['vix_reversion_signal'].fillna(0) * 0.6
            components.append(vix_signal)
        
        # Social media spike
        if 'social_mentions_spike_up' in features.columns:
            social_signal = features['social_mentions_spike_up'].fillna(0) * 0.5
            components.append(social_signal)
        
        # Sentiment bullish divergence
        if 'sentiment_bullish_divergence' in features.columns:
            div_signal = features['sentiment_bullish_divergence'].fillna(0) * 0.6
            components.append(div_signal)
        
        if components:
            signal = pd.concat(components, axis=1).max(axis=1)  # Use max for sentiment
        
        return signal.fillna(0).clip(0, 1)
    
    def _get_technical_signal(self, features: pd.DataFrame) -> pd.Series:
        """Extract technical indicator confirmation signal."""
        signal = pd.Series(0.0, index=features.index)
        
        components = []
        
        # RSI oversold
        if 'RSI_oversold' in features.columns:
            rsi_signal = features['RSI_oversold'].fillna(0) * 0.7
            components.append(rsi_signal)
        
        # MACD bullish cross
        if 'MACD_bullish_cross' in features.columns:
            macd_signal = features['MACD_bullish_cross'].fillna(0) * 0.6
            components.append(macd_signal)
        
        # Stochastic oversold with bullish cross
        if 'Stoch_bullish_cross' in features.columns:
            stoch_signal = features['Stoch_bullish_cross'].fillna(0) * 0.5
            components.append(stoch_signal)
        
        # MFI oversold
        if 'MFI_oversold' in features.columns:
            mfi_signal = features['MFI_oversold'].fillna(0) * 0.5
            components.append(mfi_signal)
        
        # Price at support (Bollinger lower band)
        if 'Bollinger_lower_breakout' in features.columns:
            bb_signal = features['Bollinger_lower_breakout'].fillna(0) * 0.6
            components.append(bb_signal)
        
        # Supertrend bullish
        if 'Supertrend_direction' in features.columns:
            st_signal = (features['Supertrend_direction'] == 1).astype(float) * 0.4
            components.append(st_signal)
        
        if components:
            signal = pd.concat(components, axis=1).mean(axis=1)
        
        return signal.fillna(0).clip(0, 1)
    
    def _apply_cooldown(self, signals: pd.Series) -> pd.Series:
        """Apply cooldown period between signals."""
        cooldown = self.config.signal_cooldown
        
        result = signals.copy()
        last_signal_idx = -cooldown - 1
        
        for i in range(len(signals)):
            if signals.iloc[i] > 0:
                if i - last_signal_idx > cooldown:
                    last_signal_idx = i
                else:
                    result.iloc[i] = 0
        
        return result
    
    def get_signal_summary(self) -> Dict:
        """Get summary of generated signals."""
        if self.signals is None:
            return {}
        
        total_signals = (self.signals > 0).sum()
        strong_signals = (self.signals == 2).sum()
        regular_signals = (self.signals == 1).sum()
        
        # Signal dates
        signal_dates = self.signals[self.signals > 0].index.tolist()
        
        # Average ensemble score when signal fires
        if self.all_features is not None and 'ensemble_score' in self.all_features.columns:
            avg_score_at_signal = self.all_features.loc[
                self.signals > 0, 'ensemble_score'
            ].mean()
        else:
            avg_score_at_signal = None
        
        return {
            'total_signals': total_signals,
            'strong_signals': strong_signals,
            'regular_signals': regular_signals,
            'signal_dates': signal_dates[-10:],  # Last 10 signals
            'avg_score_at_signal': avg_score_at_signal,
            'signal_rate': total_signals / len(self.signals) if len(self.signals) > 0 else 0
        }


class RallyPredictor:
    """
    Specialized predictor for detecting early signs of stock rallies.
    Combines multiple signals with specific emphasis on pre-rally patterns.
    """
    
    def __init__(self):
        self.ensemble = EnsembleSignalGenerator()
        
    def predict_rallies(self, data: pd.DataFrame, 
                       rally_threshold: float = 0.15,
                       lookforward: int = 20) -> pd.DataFrame:
        """
        Predict potential rallies and validate against actual future returns.
        
        Args:
            data: OHLCV data
            rally_threshold: Minimum return to count as rally (15%)
            lookforward: Days to look forward for rally
            
        Returns:
            DataFrame with predictions and validation
        """
        # Generate signals
        features, signals = self.ensemble.generate_signals(data)
        
        # Calculate forward returns
        forward_returns = data['close'].pct_change(lookforward).shift(-lookforward)
        
        # Mark actual rallies
        actual_rallies = (forward_returns >= rally_threshold).astype(int)
        
        # Create prediction DataFrame
        predictions = pd.DataFrame({
            'signal': signals,
            'ensemble_score': features.get('ensemble_score', 0),
            'forward_return': forward_returns,
            'actual_rally': actual_rallies
        }, index=data.index)
        
        # Calculate prediction accuracy
        predictions['correct'] = (
            (predictions['signal'] > 0) & (predictions['actual_rally'] == 1)
        ).astype(int)
        
        predictions['false_positive'] = (
            (predictions['signal'] > 0) & (predictions['actual_rally'] == 0)
        ).astype(int)
        
        predictions['missed_rally'] = (
            (predictions['signal'] == 0) & (predictions['actual_rally'] == 1)
        ).astype(int)
        
        return predictions
    
    def get_prediction_metrics(self, predictions: pd.DataFrame) -> Dict:
        """Calculate prediction performance metrics."""
        total_signals = (predictions['signal'] > 0).sum()
        actual_rallies = predictions['actual_rally'].sum()
        
        if total_signals == 0:
            return {'error': 'No signals generated'}
        
        true_positives = predictions['correct'].sum()
        false_positives = predictions['false_positive'].sum()
        false_negatives = predictions['missed_rally'].sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average return when signal fires
        avg_return_on_signal = predictions.loc[
            predictions['signal'] > 0, 'forward_return'
        ].mean()
        
        return {
            'total_signals': total_signals,
            'actual_rallies': actual_rallies,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_return_on_signal': avg_return_on_signal
        }


if __name__ == "__main__":
    import yfinance as yf
    
    # Fetch RKLB data
    print("Fetching RKLB data...")
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    
    # Generate signals
    print("Generating ensemble signals...")
    generator = EnsembleSignalGenerator()
    features, signals = generator.generate_signals(data)
    
    # Print summary
    summary = generator.get_signal_summary()
    print("\n=== Signal Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Rally prediction
    print("\n=== Rally Prediction ===")
    predictor = RallyPredictor()
    predictions = predictor.predict_rallies(data)
    metrics = predictor.get_prediction_metrics(predictions)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
