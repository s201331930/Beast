"""
Statistical Anomaly Detection Module

Implements various classical statistical methods for detecting anomalies:
- Z-Score based detection (Gaussian assumption)
- Modified Z-Score (robust to outliers)
- Grubbs Test (single outlier detection)
- Generalized ESD Test (multiple outliers)
- IQR (Interquartile Range) method
- Dixon's Q Test
- Mahalanobis Distance (multivariate)
- Kolmogorov-Smirnov Test (distribution shift)
- CUSUM (Cumulative Sum Control Chart)
- Exponentially Weighted Moving Average (EWMA) Control Charts

Mathematical Foundation: Based on hypothesis testing and probability theory
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class StatisticalAnomalyDetector:
    """
    Comprehensive statistical anomaly detection using classical methods.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price/volume dataframe.
        
        Args:
            df: DataFrame with columns like 'close', 'returns', 'volume', etc.
        """
        self.df = df.copy()
        self.anomalies = pd.DataFrame(index=df.index)
        
    def zscore_anomaly(self, 
                       column: str = 'returns',
                       window: int = 20,
                       threshold: float = None) -> pd.Series:
        """
        Detect anomalies using rolling Z-score.
        
        Mathematical basis: Z = (x - μ) / σ
        Assumes approximate normality of returns.
        
        Args:
            column: Column to analyze
            window: Rolling window for mean/std calculation
            threshold: Z-score threshold (default from config)
            
        Returns:
            Series of Z-scores
        """
        threshold = threshold or config.anomaly.zscore_threshold
        
        series = self.df[column]
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        zscore = (series - rolling_mean) / rolling_std
        
        self.anomalies[f'{column}_zscore'] = zscore
        self.anomalies[f'{column}_zscore_anomaly'] = np.abs(zscore) > threshold
        
        return zscore
    
    def modified_zscore_anomaly(self,
                                column: str = 'returns',
                                window: int = 20,
                                threshold: float = 3.5) -> pd.Series:
        """
        Modified Z-score using median and MAD (Median Absolute Deviation).
        More robust to outliers than standard Z-score.
        
        Mathematical basis: M = 0.6745 * (x - median) / MAD
        The constant 0.6745 makes MAD consistent with std for normal distribution.
        
        Args:
            column: Column to analyze
            window: Rolling window
            threshold: Modified Z-score threshold (typically 3.5)
            
        Returns:
            Series of modified Z-scores
        """
        series = self.df[column]
        
        # Rolling median
        rolling_median = series.rolling(window=window).median()
        
        # Rolling MAD (Median Absolute Deviation)
        def rolling_mad(x):
            median = np.median(x)
            return np.median(np.abs(x - median))
        
        rolling_mad_series = series.rolling(window=window).apply(rolling_mad, raw=True)
        
        # Modified Z-score
        modified_zscore = 0.6745 * (series - rolling_median) / rolling_mad_series
        
        self.anomalies[f'{column}_mod_zscore'] = modified_zscore
        self.anomalies[f'{column}_mod_zscore_anomaly'] = np.abs(modified_zscore) > threshold
        
        return modified_zscore
    
    def grubbs_test(self,
                    column: str = 'returns',
                    window: int = 50,
                    alpha: float = None) -> pd.Series:
        """
        Grubbs' Test for detecting single outliers in a dataset.
        
        Mathematical basis:
        G = max|x_i - x̄| / s
        
        Critical value from t-distribution:
        G_critical = ((n-1)/√n) * √(t²_{α/(2n),n-2} / (n-2+t²_{α/(2n),n-2}))
        
        Args:
            column: Column to analyze
            window: Window size for test
            alpha: Significance level
            
        Returns:
            Series indicating Grubbs test results
        """
        alpha = alpha or config.anomaly.grubbs_alpha
        series = self.df[column]
        
        def grubbs_statistic(x):
            n = len(x)
            if n < 3:
                return 0
            
            mean = np.mean(x)
            std = np.std(x, ddof=1)
            
            if std == 0:
                return 0
            
            # Maximum deviation
            max_dev = np.max(np.abs(x - mean))
            G = max_dev / std
            
            # Critical value
            t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
            
            return 1 if G > G_critical else 0
        
        grubbs_result = series.rolling(window=window).apply(
            lambda x: grubbs_statistic(x.values), raw=False
        )
        
        self.anomalies[f'{column}_grubbs'] = grubbs_result
        
        return grubbs_result
    
    def generalized_esd_test(self,
                             column: str = 'returns',
                             window: int = 50,
                             max_outliers: int = 5,
                             alpha: float = 0.05) -> pd.Series:
        """
        Generalized Extreme Studentized Deviate (ESD) test.
        Can detect multiple outliers in a dataset.
        
        This is a sequential application of the Grubbs test.
        """
        series = self.df[column]
        
        def esd_test(x, max_outliers=max_outliers, alpha=alpha):
            x = np.array(x)
            n = len(x)
            
            if n < 3:
                return 0
            
            outlier_count = 0
            
            for i in range(min(max_outliers, n - 2)):
                mean = np.mean(x)
                std = np.std(x, ddof=1)
                
                if std == 0:
                    break
                
                # Find maximum deviation
                deviations = np.abs(x - mean)
                max_idx = np.argmax(deviations)
                R = deviations[max_idx] / std
                
                # Critical value
                p = 1 - alpha / (2 * (n - i))
                t_critical = stats.t.ppf(p, n - i - 2)
                lambda_critical = (n - i - 1) * t_critical / np.sqrt((n - i - 2 + t_critical**2) * (n - i))
                
                if R > lambda_critical:
                    outlier_count += 1
                    x = np.delete(x, max_idx)
                else:
                    break
            
            return outlier_count
        
        esd_result = series.rolling(window=window).apply(esd_test, raw=True)
        
        self.anomalies[f'{column}_esd_outliers'] = esd_result
        self.anomalies[f'{column}_esd_anomaly'] = esd_result > 0
        
        return esd_result
    
    def iqr_anomaly(self,
                    column: str = 'returns',
                    window: int = 20,
                    multiplier: float = None) -> pd.Series:
        """
        Interquartile Range (IQR) based anomaly detection.
        
        Mathematical basis:
        IQR = Q3 - Q1
        Lower bound = Q1 - k * IQR
        Upper bound = Q3 + k * IQR
        
        Non-parametric method, doesn't assume normality.
        
        Args:
            column: Column to analyze
            window: Rolling window
            multiplier: IQR multiplier (default 1.5)
            
        Returns:
            Series of IQR position (how far outside bounds)
        """
        multiplier = multiplier or config.anomaly.iqr_multiplier
        series = self.df[column]
        
        q1 = series.rolling(window=window).quantile(0.25)
        q3 = series.rolling(window=window).quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Calculate how far outside bounds (0 if within bounds)
        iqr_position = pd.Series(0.0, index=series.index)
        iqr_position = np.where(series < lower_bound, (lower_bound - series) / iqr, iqr_position)
        iqr_position = np.where(series > upper_bound, (series - upper_bound) / iqr, iqr_position)
        iqr_position = pd.Series(iqr_position, index=series.index)
        
        self.anomalies[f'{column}_iqr_position'] = iqr_position
        self.anomalies[f'{column}_iqr_anomaly'] = iqr_position > 0
        
        return iqr_position
    
    def mahalanobis_distance(self,
                             columns: List[str] = None,
                             window: int = 60) -> pd.Series:
        """
        Multivariate anomaly detection using Mahalanobis distance.
        
        Mathematical basis:
        D_M = √[(x-μ)ᵀ Σ⁻¹ (x-μ)]
        
        Where Σ is the covariance matrix.
        Accounts for correlations between variables.
        
        Args:
            columns: List of columns for multivariate analysis
            window: Rolling window for covariance estimation
            
        Returns:
            Series of Mahalanobis distances
        """
        if columns is None:
            columns = ['returns', 'volume_change', 'high_low_range']
        
        # Filter to available columns
        columns = [c for c in columns if c in self.df.columns]
        
        if len(columns) < 2:
            return pd.Series(0, index=self.df.index)
        
        data = self.df[columns].dropna()
        
        distances = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            current_point = data.iloc[i].values.reshape(1, -1)
            
            mean = window_data.mean().values
            cov = window_data.cov().values
            
            try:
                # Add small regularization for numerical stability
                cov_inv = np.linalg.inv(cov + np.eye(len(columns)) * 1e-6)
                diff = current_point - mean
                mahal_dist = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))[0, 0]
                distances.iloc[i] = mahal_dist
            except:
                distances.iloc[i] = np.nan
        
        # Chi-squared threshold for p-value
        threshold = stats.chi2.ppf(0.95, df=len(columns))
        
        self.anomalies['mahalanobis_dist'] = distances
        self.anomalies['mahalanobis_anomaly'] = distances > threshold
        
        return distances
    
    def cusum_detection(self,
                        column: str = 'returns',
                        threshold: float = 4.0,
                        drift: float = 0.5) -> Tuple[pd.Series, pd.Series]:
        """
        CUSUM (Cumulative Sum Control Chart) for change detection.
        
        Mathematical basis:
        S⁺ₙ = max(0, S⁺ₙ₋₁ + (xₙ - μ - k))
        S⁻ₙ = min(0, S⁻ₙ₋₁ + (xₙ - μ + k))
        
        Detects shifts in mean level.
        
        Args:
            column: Column to analyze
            threshold: Detection threshold (h parameter)
            drift: Allowable drift (k parameter, typically σ/2)
            
        Returns:
            Tuple of (positive CUSUM, negative CUSUM)
        """
        series = self.df[column].dropna()
        
        # Standardize
        mean = series.rolling(window=50).mean()
        std = series.rolling(window=50).std()
        standardized = (series - mean) / std
        
        # Initialize CUSUM
        cusum_pos = pd.Series(0.0, index=series.index)
        cusum_neg = pd.Series(0.0, index=series.index)
        
        for i in range(1, len(series)):
            cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + standardized.iloc[i] - drift)
            cusum_neg.iloc[i] = min(0, cusum_neg.iloc[i-1] + standardized.iloc[i] + drift)
        
        self.anomalies[f'{column}_cusum_pos'] = cusum_pos
        self.anomalies[f'{column}_cusum_neg'] = cusum_neg
        self.anomalies[f'{column}_cusum_anomaly'] = (cusum_pos > threshold) | (cusum_neg < -threshold)
        
        return cusum_pos, cusum_neg
    
    def ewma_control_chart(self,
                           column: str = 'returns',
                           lambda_param: float = 0.2,
                           L: float = 3.0) -> pd.Series:
        """
        EWMA (Exponentially Weighted Moving Average) Control Chart.
        
        Mathematical basis:
        Z_t = λ * x_t + (1-λ) * Z_{t-1}
        Control limits: μ ± L * σ * √(λ/(2-λ) * [1-(1-λ)^(2t)])
        
        Good for detecting small shifts in the mean.
        
        Args:
            column: Column to analyze
            lambda_param: Smoothing parameter (0 < λ ≤ 1)
            L: Width of control limits in standard deviations
            
        Returns:
            Series of EWMA statistics
        """
        series = self.df[column].dropna()
        
        # Historical parameters
        mu = series.rolling(window=50).mean()
        sigma = series.rolling(window=50).std()
        
        # Initialize EWMA
        ewma = pd.Series(index=series.index, dtype=float)
        ewma.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            ewma.iloc[i] = lambda_param * series.iloc[i] + (1 - lambda_param) * ewma.iloc[i-1]
        
        # Control limits
        ewma_std = sigma * np.sqrt(lambda_param / (2 - lambda_param))
        upper_cl = mu + L * ewma_std
        lower_cl = mu - L * ewma_std
        
        # Standardized EWMA
        ewma_zscore = (ewma - mu) / ewma_std
        
        self.anomalies[f'{column}_ewma'] = ewma
        self.anomalies[f'{column}_ewma_zscore'] = ewma_zscore
        self.anomalies[f'{column}_ewma_anomaly'] = (ewma > upper_cl) | (ewma < lower_cl)
        
        return ewma_zscore
    
    def kolmogorov_smirnov_shift(self,
                                  column: str = 'returns',
                                  window: int = 50,
                                  reference_window: int = 100) -> pd.Series:
        """
        Kolmogorov-Smirnov test for distribution shift detection.
        
        Mathematical basis:
        D = sup_x |F_1(x) - F_2(x)|
        
        Detects changes in the underlying distribution.
        
        Args:
            column: Column to analyze
            window: Current window size
            reference_window: Reference/baseline window size
            
        Returns:
            Series of KS statistics (p-values)
        """
        series = self.df[column].dropna()
        
        ks_stats = pd.Series(index=series.index, dtype=float)
        ks_pvalues = pd.Series(index=series.index, dtype=float)
        
        total_window = window + reference_window
        
        for i in range(total_window, len(series)):
            reference = series.iloc[i-total_window:i-window].values
            current = series.iloc[i-window:i].values
            
            stat, pvalue = stats.ks_2samp(reference, current)
            ks_stats.iloc[i] = stat
            ks_pvalues.iloc[i] = pvalue
        
        self.anomalies[f'{column}_ks_stat'] = ks_stats
        self.anomalies[f'{column}_ks_pvalue'] = ks_pvalues
        self.anomalies[f'{column}_ks_anomaly'] = ks_pvalues < 0.05  # Distribution shift detected
        
        return ks_pvalues
    
    def volume_anomaly(self,
                       window: int = 20,
                       threshold: float = None) -> pd.Series:
        """
        Detect volume anomalies using multiple methods.
        
        Volume spikes often precede or accompany significant price moves.
        
        Args:
            window: Rolling window
            threshold: Z-score threshold
            
        Returns:
            Combined volume anomaly score
        """
        threshold = threshold or config.anomaly.volume_zscore_threshold
        
        volume = self.df['volume']
        
        # Z-score
        vol_mean = volume.rolling(window=window).mean()
        vol_std = volume.rolling(window=window).std()
        vol_zscore = (volume - vol_mean) / vol_std
        
        # Volume ratio
        vol_ratio = volume / vol_mean
        
        # Percentile rank
        vol_percentile = volume.rolling(window=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 20 else 0.5,
            raw=False
        )
        
        # Combined score
        vol_anomaly_score = (
            (vol_zscore / threshold).clip(-1, 2) * 0.4 +
            ((vol_ratio - 1) / (config.anomaly.volume_ratio_threshold - 1)).clip(-1, 2) * 0.3 +
            (vol_percentile - 0.5) * 2 * 0.3
        )
        
        self.anomalies['volume_zscore'] = vol_zscore
        self.anomalies['volume_ratio'] = vol_ratio
        self.anomalies['volume_percentile'] = vol_percentile
        self.anomalies['volume_anomaly_score'] = vol_anomaly_score
        self.anomalies['volume_anomaly'] = vol_anomaly_score > 0.5
        
        return vol_anomaly_score
    
    def price_gap_anomaly(self, threshold: float = 0.02) -> pd.Series:
        """
        Detect price gap anomalies (gaps between close and next open).
        
        Large gaps often indicate overnight news or events.
        
        Args:
            threshold: Minimum gap percentage to flag
            
        Returns:
            Series of gap anomaly indicators
        """
        if 'gap' not in self.df.columns:
            self.df['gap'] = (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)
        
        gap = self.df['gap']
        
        # Gap z-score
        gap_mean = gap.rolling(window=50).mean()
        gap_std = gap.rolling(window=50).std()
        gap_zscore = (gap - gap_mean) / gap_std
        
        self.anomalies['gap'] = gap
        self.anomalies['gap_zscore'] = gap_zscore
        self.anomalies['gap_anomaly'] = np.abs(gap) > threshold
        
        return gap_zscore
    
    def run_all_detectors(self) -> pd.DataFrame:
        """
        Run all statistical anomaly detectors and return combined results.
        """
        print("Running statistical anomaly detection...")
        
        # Z-score methods
        self.zscore_anomaly('returns')
        self.modified_zscore_anomaly('returns')
        
        # Test-based methods
        self.grubbs_test('returns')
        self.generalized_esd_test('returns')
        
        # IQR method
        self.iqr_anomaly('returns')
        
        # Multivariate
        self.mahalanobis_distance()
        
        # Change detection
        self.cusum_detection('returns')
        self.ewma_control_chart('returns')
        
        # Distribution shift
        self.kolmogorov_smirnov_shift('returns')
        
        # Volume and gaps
        self.volume_anomaly()
        self.price_gap_anomaly()
        
        # Aggregate statistical anomaly score
        anomaly_columns = [c for c in self.anomalies.columns if c.endswith('_anomaly')]
        self.anomalies['stat_anomaly_count'] = self.anomalies[anomaly_columns].sum(axis=1)
        self.anomalies['stat_anomaly_ratio'] = self.anomalies['stat_anomaly_count'] / len(anomaly_columns)
        
        print(f"  Detected anomalies using {len(anomaly_columns)} methods")
        print(f"  Total anomaly flags: {self.anomalies['stat_anomaly_count'].sum()}")
        
        return self.anomalies
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary statistics of detected anomalies."""
        anomaly_cols = [c for c in self.anomalies.columns if c.endswith('_anomaly')]
        
        summary = {
            'total_observations': len(self.anomalies),
            'methods_used': len(anomaly_cols),
            'anomalies_per_method': {
                col.replace('_anomaly', ''): self.anomalies[col].sum()
                for col in anomaly_cols
            },
            'avg_anomaly_ratio': self.anomalies['stat_anomaly_ratio'].mean(),
            'max_concurrent_anomalies': self.anomalies['stat_anomaly_count'].max()
        }
        
        return summary


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, '..')
    from data.collector import DataCollector
    
    # Collect data
    collector = DataCollector("RKLB")
    data = collector.collect_all_data()
    df = data['primary']
    
    # Run anomaly detection
    detector = StatisticalAnomalyDetector(df)
    anomalies = detector.run_all_detectors()
    
    # Print summary
    summary = detector.get_anomaly_summary()
    print("\nAnomaly Detection Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
