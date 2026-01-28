"""
Statistical Anomaly Detection Models
=====================================
Implements classical statistical methods for detecting anomalies in financial time series:
- Z-Score based anomaly detection
- Bollinger Bands
- GARCH volatility modeling
- Mahalanobis distance
- Grubbs' test
- Modified Z-score (MAD-based)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
from typing import Tuple, Dict, Optional, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ZScoreAnomalyDetector:
    """
    Z-Score based anomaly detection.
    Flags values that deviate significantly from the rolling mean.
    """
    
    def __init__(self, window: int = 20, threshold: float = 2.5):
        self.window = window
        self.threshold = threshold
        self.name = "ZScore"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate z-scores and detect anomalies.
        
        Returns DataFrame with:
        - zscore: Z-score values
        - anomaly: Boolean flag for anomalies
        - anomaly_direction: 1 for positive, -1 for negative anomalies
        """
        rolling_mean = series.rolling(window=self.window).mean()
        rolling_std = series.rolling(window=self.window).std()
        
        zscore = (series - rolling_mean) / rolling_std
        
        anomaly = np.abs(zscore) > self.threshold
        anomaly_direction = np.sign(zscore) * anomaly.astype(int)
        
        return pd.DataFrame({
            f'{self.name}_zscore': zscore,
            f'{self.name}_anomaly': anomaly,
            f'{self.name}_direction': anomaly_direction,
            f'{self.name}_score': np.abs(zscore) / self.threshold  # Normalized score
        }, index=series.index)
    
    def detect_returns_anomalies(self, returns: pd.Series) -> pd.DataFrame:
        """Detect anomalies in returns series."""
        return self.fit_transform(returns)
    
    def detect_volume_anomalies(self, volume: pd.Series) -> pd.DataFrame:
        """Detect anomalies in volume series."""
        log_volume = np.log(volume + 1)
        result = self.fit_transform(log_volume)
        result.columns = [col.replace('ZScore', 'ZScore_Volume') for col in result.columns]
        return result


class ModifiedZScoreDetector:
    """
    Modified Z-Score using Median Absolute Deviation (MAD).
    More robust to outliers than standard Z-score.
    """
    
    def __init__(self, window: int = 20, threshold: float = 3.5):
        self.window = window
        self.threshold = threshold
        self.name = "MAD_ZScore"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate modified z-scores using MAD.
        """
        def rolling_mad(x):
            median = np.median(x)
            mad = np.median(np.abs(x - median))
            return mad
        
        rolling_median = series.rolling(window=self.window).median()
        rolling_mad = series.rolling(window=self.window).apply(rolling_mad, raw=True)
        
        # Modified z-score: 0.6745 is the scaling constant for normal distribution
        modified_zscore = 0.6745 * (series - rolling_median) / (rolling_mad + 1e-10)
        
        anomaly = np.abs(modified_zscore) > self.threshold
        
        return pd.DataFrame({
            f'{self.name}_zscore': modified_zscore,
            f'{self.name}_anomaly': anomaly,
            f'{self.name}_score': np.abs(modified_zscore) / self.threshold
        }, index=series.index)


class BollingerBandsDetector:
    """
    Bollinger Bands based anomaly detection.
    Detects price breakouts beyond the bands.
    """
    
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        self.window = window
        self.std_dev = std_dev
        self.name = "Bollinger"
        
    def fit_transform(self, price: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and detect breakouts.
        """
        sma = price.rolling(window=self.window).mean()
        rolling_std = price.rolling(window=self.window).std()
        
        upper_band = sma + (rolling_std * self.std_dev)
        lower_band = sma - (rolling_std * self.std_dev)
        
        # %B indicator: shows where price is relative to bands
        percent_b = (price - lower_band) / (upper_band - lower_band)
        
        # Bandwidth: shows volatility
        bandwidth = (upper_band - lower_band) / sma
        
        # Anomalies: outside the bands
        upper_breakout = price > upper_band
        lower_breakout = price < lower_band
        anomaly = upper_breakout | lower_breakout
        
        # Squeeze detection (low volatility before potential move)
        bandwidth_ma = bandwidth.rolling(window=20).mean()
        squeeze = bandwidth < bandwidth_ma * 0.75
        
        return pd.DataFrame({
            f'{self.name}_sma': sma,
            f'{self.name}_upper': upper_band,
            f'{self.name}_lower': lower_band,
            f'{self.name}_percent_b': percent_b,
            f'{self.name}_bandwidth': bandwidth,
            f'{self.name}_upper_breakout': upper_breakout,
            f'{self.name}_lower_breakout': lower_breakout,
            f'{self.name}_anomaly': anomaly,
            f'{self.name}_squeeze': squeeze,
            f'{self.name}_score': np.abs(percent_b - 0.5) * 2  # Distance from middle
        }, index=price.index)


class GARCHVolatilityDetector:
    """
    GARCH(1,1) model for volatility anomaly detection.
    Detects when realized volatility significantly differs from expected.
    """
    
    def __init__(self, p: int = 1, q: int = 1, vol_threshold: float = 2.0):
        self.p = p
        self.q = q
        self.vol_threshold = vol_threshold
        self.name = "GARCH"
        
    def fit_transform(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fit GARCH model and detect volatility anomalies.
        """
        try:
            from arch import arch_model
            
            # Clean returns
            clean_returns = returns.dropna() * 100  # Scale for numerical stability
            
            if len(clean_returns) < 100:
                logger.warning("Insufficient data for GARCH model")
                return self._fallback_volatility(returns)
            
            # Fit GARCH model
            model = arch_model(clean_returns, vol='Garch', p=self.p, q=self.q)
            result = model.fit(disp='off')
            
            # Get conditional volatility
            cond_vol = result.conditional_volatility / 100  # Scale back
            
            # Realized volatility (rolling std of returns)
            realized_vol = returns.rolling(window=20).std()
            
            # Volatility surprise: realized vs expected
            vol_surprise = realized_vol / (cond_vol.reindex(returns.index) + 1e-10)
            
            # Anomaly: when volatility is significantly higher than expected
            anomaly = vol_surprise > self.vol_threshold
            
            # Standardized residuals for tail events
            std_resid = result.std_resid
            
            return pd.DataFrame({
                f'{self.name}_cond_vol': cond_vol.reindex(returns.index),
                f'{self.name}_realized_vol': realized_vol,
                f'{self.name}_vol_surprise': vol_surprise,
                f'{self.name}_anomaly': anomaly,
                f'{self.name}_std_resid': std_resid.reindex(returns.index),
                f'{self.name}_score': vol_surprise / self.vol_threshold
            }, index=returns.index)
            
        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}, using fallback")
            return self._fallback_volatility(returns)
    
    def _fallback_volatility(self, returns: pd.Series) -> pd.DataFrame:
        """Fallback method using EWMA volatility."""
        ewma_vol = returns.ewm(span=20).std()
        realized_vol = returns.rolling(20).std()
        vol_surprise = realized_vol / (ewma_vol + 1e-10)
        anomaly = vol_surprise > self.vol_threshold
        
        return pd.DataFrame({
            f'{self.name}_cond_vol': ewma_vol,
            f'{self.name}_realized_vol': realized_vol,
            f'{self.name}_vol_surprise': vol_surprise,
            f'{self.name}_anomaly': anomaly,
            f'{self.name}_score': vol_surprise / self.vol_threshold
        }, index=returns.index)


class MahalanobisDetector:
    """
    Mahalanobis distance based multivariate anomaly detection.
    Considers correlations between multiple features.
    """
    
    def __init__(self, window: int = 60, threshold_percentile: float = 95):
        self.window = window
        self.threshold_percentile = threshold_percentile
        self.name = "Mahalanobis"
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Mahalanobis distances for multivariate data.
        """
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        df = data[numeric_cols].dropna()
        
        if len(df) < self.window:
            logger.warning("Insufficient data for Mahalanobis calculation")
            return pd.DataFrame(index=data.index)
        
        # Calculate rolling Mahalanobis distance
        distances = []
        
        for i in range(self.window, len(df)):
            window_data = df.iloc[i-self.window:i]
            current_point = df.iloc[i:i+1]
            
            try:
                # Calculate covariance matrix
                cov_matrix = window_data.cov()
                cov_inv = np.linalg.pinv(cov_matrix)
                
                # Mean of window
                mean = window_data.mean()
                
                # Mahalanobis distance
                diff = current_point.values[0] - mean.values
                dist = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                distances.append(dist)
            except Exception:
                distances.append(np.nan)
        
        # Create result DataFrame
        result_index = df.index[self.window:]
        mahal_dist = pd.Series(distances, index=result_index)
        
        # Threshold based on chi-square distribution
        dof = len(numeric_cols)
        chi2_threshold = chi2.ppf(self.threshold_percentile / 100, dof)
        threshold = np.sqrt(chi2_threshold)
        
        anomaly = mahal_dist > threshold
        
        result = pd.DataFrame({
            f'{self.name}_distance': mahal_dist,
            f'{self.name}_anomaly': anomaly,
            f'{self.name}_score': mahal_dist / threshold
        }, index=result_index)
        
        return result.reindex(data.index)


class ExtremeValueDetector:
    """
    Extreme Value Theory (EVT) based tail risk detection.
    Uses Generalized Pareto Distribution for tail estimation.
    """
    
    def __init__(self, threshold_percentile: float = 95, window: int = 252):
        self.threshold_percentile = threshold_percentile
        self.window = window
        self.name = "EVT"
        
    def fit_transform(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fit GPD to tails and detect extreme events.
        """
        results = []
        
        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i-self.window:i]
            current_return = returns.iloc[i]
            
            # Upper and lower thresholds
            upper_threshold = window_returns.quantile(self.threshold_percentile / 100)
            lower_threshold = window_returns.quantile(1 - self.threshold_percentile / 100)
            
            # Check for extreme values
            is_extreme_up = current_return > upper_threshold
            is_extreme_down = current_return < lower_threshold
            
            # Calculate exceedance probability (simplified)
            if is_extreme_up:
                exceedance = (current_return - upper_threshold) / window_returns.std()
            elif is_extreme_down:
                exceedance = (lower_threshold - current_return) / window_returns.std()
            else:
                exceedance = 0
            
            results.append({
                'upper_threshold': upper_threshold,
                'lower_threshold': lower_threshold,
                'extreme_up': is_extreme_up,
                'extreme_down': is_extreme_down,
                'exceedance': exceedance
            })
        
        result_df = pd.DataFrame(results, index=returns.index[self.window:])
        result_df[f'{self.name}_anomaly'] = result_df['extreme_up'] | result_df['extreme_down']
        result_df[f'{self.name}_score'] = result_df['exceedance']
        
        return result_df.reindex(returns.index)


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change point detection.
    Detects shifts in the mean of a time series.
    """
    
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.name = "CUSUM"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Apply CUSUM algorithm to detect change points.
        """
        # Standardize series
        mean = series.rolling(window=50).mean()
        std = series.rolling(window=50).std()
        standardized = (series - mean) / (std + 1e-10)
        
        # Initialize CUSUM statistics
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        
        for i in range(1, len(series)):
            if not np.isnan(standardized.iloc[i]):
                cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i] - self.drift)
                cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i] + self.drift)
        
        # Detect anomalies
        anomaly_up = cusum_pos > self.threshold
        anomaly_down = cusum_neg < -self.threshold
        
        return pd.DataFrame({
            f'{self.name}_pos': cusum_pos,
            f'{self.name}_neg': cusum_neg,
            f'{self.name}_anomaly_up': anomaly_up,
            f'{self.name}_anomaly_down': anomaly_down,
            f'{self.name}_anomaly': anomaly_up | anomaly_down,
            f'{self.name}_score': np.maximum(cusum_pos / self.threshold, 
                                              np.abs(cusum_neg) / self.threshold)
        }, index=series.index)


class StatisticalAnomalyEnsemble:
    """
    Ensemble of all statistical anomaly detection methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.detectors = self._initialize_detectors()
        
    def _initialize_detectors(self) -> Dict:
        """Initialize all statistical detectors."""
        return {
            'zscore': ZScoreAnomalyDetector(
                window=self.config.get('zscore_window', 20),
                threshold=self.config.get('zscore_threshold', 2.5)
            ),
            'mad_zscore': ModifiedZScoreDetector(
                window=self.config.get('mad_window', 20),
                threshold=self.config.get('mad_threshold', 3.5)
            ),
            'bollinger': BollingerBandsDetector(
                window=self.config.get('bollinger_window', 20),
                std_dev=self.config.get('bollinger_std', 2.0)
            ),
            'garch': GARCHVolatilityDetector(
                p=self.config.get('garch_p', 1),
                q=self.config.get('garch_q', 1)
            ),
            'mahalanobis': MahalanobisDetector(
                window=self.config.get('mahal_window', 60)
            ),
            'evt': ExtremeValueDetector(
                threshold_percentile=self.config.get('evt_percentile', 95)
            ),
            'cusum': CUSUMDetector(
                threshold=self.config.get('cusum_threshold', 5.0)
            )
        }
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all statistical anomaly detectors to the data.
        """
        results = {}
        
        # Extract key series
        returns = data['returns'] if 'returns' in data.columns else data['close'].pct_change()
        price = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        volume = data['volume'] if 'volume' in data.columns else None
        
        logger.info("Running statistical anomaly detection ensemble...")
        
        # Z-Score on returns
        results['zscore_returns'] = self.detectors['zscore'].fit_transform(returns)
        
        # Z-Score on volume
        if volume is not None:
            results['zscore_volume'] = self.detectors['zscore'].detect_volume_anomalies(volume)
        
        # Modified Z-Score
        results['mad'] = self.detectors['mad_zscore'].fit_transform(returns)
        
        # Bollinger Bands
        results['bollinger'] = self.detectors['bollinger'].fit_transform(price)
        
        # GARCH
        results['garch'] = self.detectors['garch'].fit_transform(returns)
        
        # Mahalanobis (on multiple features)
        feature_cols = ['returns', 'volume_ratio', 'volatility_20d']
        feature_cols = [c for c in feature_cols if c in data.columns]
        if len(feature_cols) >= 2:
            results['mahalanobis'] = self.detectors['mahalanobis'].fit_transform(data[feature_cols])
        
        # EVT
        results['evt'] = self.detectors['evt'].fit_transform(returns)
        
        # CUSUM
        results['cusum'] = self.detectors['cusum'].fit_transform(returns)
        
        # Combine all results
        combined = pd.concat([df for df in results.values()], axis=1)
        
        # Calculate ensemble anomaly score
        anomaly_cols = [col for col in combined.columns if '_anomaly' in col and 'direction' not in col]
        score_cols = [col for col in combined.columns if '_score' in col]
        
        combined['stat_anomaly_count'] = combined[anomaly_cols].sum(axis=1)
        combined['stat_anomaly_mean_score'] = combined[score_cols].mean(axis=1)
        combined['stat_ensemble_signal'] = (combined['stat_anomaly_count'] >= 3).astype(int)
        
        logger.info(f"Statistical ensemble complete: {len(combined.columns)} features generated")
        
        return combined


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Fetch RKLB data
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    data['returns'] = data['close'].pct_change()
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['volatility_20d'] = data['returns'].rolling(20).std()
    
    # Run ensemble
    ensemble = StatisticalAnomalyEnsemble()
    results = ensemble.fit_transform(data)
    
    print("Statistical Anomaly Results:")
    print(results.tail(20))
    
    # Show anomaly summary
    anomaly_days = results[results['stat_ensemble_signal'] == 1]
    print(f"\nTotal ensemble anomaly signals: {len(anomaly_days)}")
