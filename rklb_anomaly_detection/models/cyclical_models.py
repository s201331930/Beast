"""
Cyclical and Mean Reversion Models
===================================
Implements physics and mathematics-inspired models for detecting cycles and mean reversion:
- Fourier Transform Analysis (spectral decomposition)
- Hurst Exponent (persistence/anti-persistence)
- Ornstein-Uhlenbeck Process (mean reversion speed)
- Hidden Markov Models (regime detection)
- Wavelet Analysis
- Hilbert Transform (instantaneous phase/frequency)
- Kalman Filter (state estimation)
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class FourierAnalyzer:
    """
    Fourier Transform analysis for detecting dominant cycles in price data.
    Uses spectral decomposition to identify periodic patterns.
    """
    
    def __init__(self, n_harmonics: int = 5, min_period: int = 5, max_period: int = 252):
        self.n_harmonics = n_harmonics
        self.min_period = min_period
        self.max_period = max_period
        self.name = "Fourier"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Perform Fourier analysis and detect cyclical anomalies.
        """
        # Clean and detrend series
        clean_series = series.dropna()
        detrended = self._detrend(clean_series)
        
        # Compute FFT
        n = len(detrended)
        fft_values = fft(detrended.values)
        frequencies = fftfreq(n)
        
        # Power spectrum
        power = np.abs(fft_values) ** 2
        
        # Find dominant frequencies (positive frequencies only)
        pos_mask = frequencies > 0
        pos_freq = frequencies[pos_mask]
        pos_power = power[pos_mask]
        
        # Convert to periods
        periods = 1 / (pos_freq + 1e-10)
        
        # Filter valid periods
        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)
        valid_periods = periods[valid_mask]
        valid_power = pos_power[valid_mask]
        
        # Find top N harmonics
        if len(valid_power) > 0:
            top_indices = np.argsort(valid_power)[-self.n_harmonics:]
            dominant_periods = valid_periods[top_indices]
            dominant_power = valid_power[top_indices]
        else:
            dominant_periods = np.array([20, 40, 60])  # Default periods
            dominant_power = np.array([1, 1, 1])
        
        # Reconstruct signal with dominant harmonics
        reconstructed = self._reconstruct_signal(detrended, fft_values, self.n_harmonics * 2)
        
        # Calculate phase and cycle position
        cycle_results = self._calculate_cycle_position(clean_series, dominant_periods[0] if len(dominant_periods) > 0 else 20)
        
        # Spectral entropy (measure of randomness)
        spectral_entropy = self._spectral_entropy(pos_power)
        
        # Anomaly: when actual deviates significantly from reconstructed
        residual = detrended - reconstructed
        residual_zscore = (residual - residual.mean()) / (residual.std() + 1e-10)
        anomaly = np.abs(residual_zscore) > 2.5
        
        result = pd.DataFrame({
            f'{self.name}_reconstructed': reconstructed,
            f'{self.name}_residual': residual,
            f'{self.name}_residual_zscore': residual_zscore,
            f'{self.name}_spectral_entropy': spectral_entropy,
            f'{self.name}_dominant_period': dominant_periods[0] if len(dominant_periods) > 0 else np.nan,
            f'{self.name}_cycle_phase': cycle_results['phase'],
            f'{self.name}_cycle_position': cycle_results['position'],
            f'{self.name}_anomaly': anomaly.astype(int),
            f'{self.name}_score': np.abs(residual_zscore) / 2.5
        }, index=clean_series.index)
        
        return result.reindex(series.index)
    
    def _detrend(self, series: pd.Series) -> pd.Series:
        """Remove linear trend from series."""
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values, 1)
        trend = slope * x + intercept
        return pd.Series(series.values - trend, index=series.index)
    
    def _reconstruct_signal(self, series: pd.Series, fft_values: np.ndarray, 
                           n_components: int) -> np.ndarray:
        """Reconstruct signal using top N Fourier components."""
        n = len(series)
        power = np.abs(fft_values) ** 2
        
        # Keep only top N components
        threshold_idx = np.argsort(power)[-n_components:]
        filtered_fft = np.zeros_like(fft_values)
        filtered_fft[threshold_idx] = fft_values[threshold_idx]
        
        # Inverse FFT
        reconstructed = np.real(ifft(filtered_fft))
        return reconstructed
    
    def _calculate_cycle_position(self, series: pd.Series, period: float) -> Dict:
        """Calculate position within the dominant cycle."""
        # Use Hilbert transform for instantaneous phase
        analytic_signal = signal.hilbert(series.values)
        phase = np.angle(analytic_signal)
        
        # Normalize phase to [0, 1]
        position = (phase + np.pi) / (2 * np.pi)
        
        return {'phase': phase, 'position': position}
    
    def _spectral_entropy(self, power: np.ndarray) -> float:
        """Calculate spectral entropy (randomness measure)."""
        # Normalize power
        power_norm = power / (power.sum() + 1e-10)
        # Calculate entropy
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        # Normalize by maximum entropy
        max_entropy = np.log2(len(power))
        return entropy / max_entropy if max_entropy > 0 else 0


class HurstExponentAnalyzer:
    """
    Hurst Exponent analysis for detecting mean reversion vs trending behavior.
    
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk
    H > 0.5: Trending (persistent)
    """
    
    def __init__(self, min_window: int = 20, max_window: int = 200):
        self.min_window = min_window
        self.max_window = max_window
        self.name = "Hurst"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate rolling Hurst exponent.
        """
        clean_series = series.dropna()
        
        if len(clean_series) < self.max_window:
            logger.warning("Insufficient data for Hurst calculation")
            return pd.DataFrame(index=series.index)
        
        # Calculate rolling Hurst exponent
        hurst_values = []
        for i in range(self.max_window, len(clean_series)):
            window_data = clean_series.iloc[i-self.max_window:i]
            H = self._calculate_hurst(window_data.values)
            hurst_values.append(H)
        
        # Create result series
        hurst_series = pd.Series(
            hurst_values, 
            index=clean_series.index[self.max_window:]
        )
        
        # Mean reversion signal: H < 0.5 indicates mean reversion
        mean_reversion = hurst_series < 0.5
        trending = hurst_series > 0.6
        
        # Anomaly: strong mean reversion or trending regime
        anomaly = (hurst_series < 0.35) | (hurst_series > 0.65)
        
        # Calculate confidence based on distance from 0.5
        confidence = np.abs(hurst_series - 0.5) * 2
        
        result = pd.DataFrame({
            f'{self.name}_exponent': hurst_series,
            f'{self.name}_mean_reverting': mean_reversion.astype(int),
            f'{self.name}_trending': trending.astype(int),
            f'{self.name}_anomaly': anomaly.astype(int),
            f'{self.name}_confidence': confidence,
            f'{self.name}_score': confidence
        }, index=hurst_series.index)
        
        return result.reindex(series.index)
    
    def _calculate_hurst(self, ts: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        """
        n = len(ts)
        if n < 20:
            return 0.5
        
        # Range of lags
        lags = range(2, min(n // 2, 100))
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lags:
            # Divide into sub-series
            n_subseries = n // lag
            if n_subseries < 2:
                continue
            
            rs_list = []
            for i in range(n_subseries):
                subseries = ts[i * lag:(i + 1) * lag]
                
                # Mean-adjusted series
                mean_adj = subseries - subseries.mean()
                
                # Cumulative deviation
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = cumsum.max() - cumsum.min()
                
                # Standard deviation
                S = subseries.std()
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))
        
        if len(rs_values) < 5:
            return 0.5
        
        # Linear regression in log-log space
        lags_arr = np.array([x[0] for x in rs_values])
        rs_arr = np.array([x[1] for x in rs_values])
        
        # Filter out invalid values
        valid_mask = (rs_arr > 0) & np.isfinite(rs_arr)
        if valid_mask.sum() < 5:
            return 0.5
        
        log_lags = np.log(lags_arr[valid_mask])
        log_rs = np.log(rs_arr[valid_mask])
        
        # Hurst exponent is the slope
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        return np.clip(slope, 0, 1)


class OrnsteinUhlenbeckAnalyzer:
    """
    Ornstein-Uhlenbeck process for mean reversion modeling.
    
    dX = θ(μ - X)dt + σdW
    
    Where:
    - θ is the speed of mean reversion
    - μ is the long-term mean
    - σ is volatility
    """
    
    def __init__(self, window: int = 60):
        self.window = window
        self.name = "OU"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Fit OU process and calculate mean reversion signals.
        """
        clean_series = series.dropna()
        
        if len(clean_series) < self.window:
            logger.warning("Insufficient data for OU process")
            return pd.DataFrame(index=series.index)
        
        # Rolling OU parameter estimation
        results = []
        
        for i in range(self.window, len(clean_series)):
            window_data = clean_series.iloc[i-self.window:i].values
            params = self._estimate_ou_params(window_data)
            
            current_price = clean_series.iloc[i]
            
            # Calculate expected reversion
            expected_change = params['theta'] * (params['mu'] - current_price)
            
            # Distance from mean in standard deviations
            distance_from_mean = (current_price - params['mu']) / (params['sigma'] + 1e-10)
            
            # Half-life of mean reversion (in days)
            half_life = np.log(2) / params['theta'] if params['theta'] > 0 else np.inf
            
            results.append({
                'theta': params['theta'],
                'mu': params['mu'],
                'sigma': params['sigma'],
                'expected_change': expected_change,
                'distance_from_mean': distance_from_mean,
                'half_life': half_life
            })
        
        result_df = pd.DataFrame(results, index=clean_series.index[self.window:])
        
        # Anomaly signals
        result_df[f'{self.name}_anomaly'] = (
            (np.abs(result_df['distance_from_mean']) > 2) & 
            (result_df['half_life'] < 30)
        ).astype(int)
        
        # Strong mean reversion signal
        result_df[f'{self.name}_strong_reversion'] = (
            (np.abs(result_df['distance_from_mean']) > 2.5) & 
            (result_df['half_life'] < 20)
        ).astype(int)
        
        # Direction of expected reversion
        result_df[f'{self.name}_reversion_direction'] = np.sign(result_df['expected_change'])
        
        # Normalized score
        result_df[f'{self.name}_score'] = np.abs(result_df['distance_from_mean']) / 3
        
        # Rename columns
        result_df = result_df.rename(columns={
            'theta': f'{self.name}_theta',
            'mu': f'{self.name}_mu',
            'sigma': f'{self.name}_sigma',
            'expected_change': f'{self.name}_expected_change',
            'distance_from_mean': f'{self.name}_distance_from_mean',
            'half_life': f'{self.name}_half_life'
        })
        
        return result_df.reindex(series.index)
    
    def _estimate_ou_params(self, data: np.ndarray, dt: float = 1/252) -> Dict:
        """
        Estimate OU parameters using maximum likelihood.
        """
        n = len(data)
        if n < 10:
            return {'theta': 0.1, 'mu': data.mean(), 'sigma': data.std()}
        
        # Simple regression-based estimation
        # X_t - X_{t-1} = θ(μ - X_{t-1})dt + noise
        y = np.diff(data)
        x = data[:-1]
        
        # Regression: y = a + b*x
        x_mean = x.mean()
        y_mean = y.mean()
        
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)
        
        if var > 0:
            b = cov / var
            a = y_mean - b * x_mean
            
            # θ = -b/dt, μ = -a/(b*dt)
            theta = max(-b / dt, 0.01)  # Ensure positive
            mu = -a / (b + 1e-10) if abs(b) > 1e-10 else data.mean()
        else:
            theta = 0.1
            mu = data.mean()
        
        # Estimate sigma from residuals
        predicted = a + b * x
        residuals = y - predicted
        sigma = residuals.std() / np.sqrt(dt)
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma
        }


class HiddenMarkovModelAnalyzer:
    """
    Hidden Markov Model for regime detection.
    Identifies different market states (e.g., bull, bear, sideways).
    """
    
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.name = "HMM"
        self.model = None
        
    def fit_transform(self, returns: pd.Series) -> pd.DataFrame:
        """
        Fit HMM and identify regimes.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            
            clean_returns = returns.dropna()
            
            if len(clean_returns) < 100:
                logger.warning("Insufficient data for HMM")
                return self._fallback_regime(returns)
            
            # Prepare data
            X = clean_returns.values.reshape(-1, 1)
            
            # Fit HMM
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type='full',
                n_iter=self.n_iter,
                random_state=42
            )
            
            self.model.fit(X)
            
            # Predict states
            hidden_states = self.model.predict(X)
            state_probs = self.model.predict_proba(X)
            
            # Identify regime characteristics
            state_means = self.model.means_.flatten()
            state_vars = self.model.covars_.flatten()
            
            # Sort states by mean return (bearish to bullish)
            sorted_idx = np.argsort(state_means)
            
            # Map to regime labels
            regime_map = {sorted_idx[0]: 'bear', 
                         sorted_idx[1]: 'neutral', 
                         sorted_idx[2]: 'bull'}
            
            regimes = [regime_map.get(s, 'neutral') for s in hidden_states]
            
            # Calculate regime change signals
            regime_change = np.diff(hidden_states, prepend=hidden_states[0]) != 0
            
            # Maximum probability (confidence)
            max_prob = state_probs.max(axis=1)
            
            # Anomaly: regime changes or low confidence
            anomaly = regime_change | (max_prob < 0.6)
            
            result = pd.DataFrame({
                f'{self.name}_state': hidden_states,
                f'{self.name}_regime': regimes,
                f'{self.name}_bull_prob': state_probs[:, sorted_idx[2]],
                f'{self.name}_bear_prob': state_probs[:, sorted_idx[0]],
                f'{self.name}_confidence': max_prob,
                f'{self.name}_regime_change': regime_change.astype(int),
                f'{self.name}_anomaly': anomaly.astype(int),
                f'{self.name}_score': 1 - max_prob
            }, index=clean_returns.index)
            
            return result.reindex(returns.index)
            
        except ImportError:
            logger.warning("hmmlearn not available, using fallback")
            return self._fallback_regime(returns)
    
    def _fallback_regime(self, returns: pd.Series) -> pd.DataFrame:
        """Simple momentum-based regime detection fallback."""
        clean_returns = returns.dropna()
        
        # Rolling mean and volatility
        rolling_mean = clean_returns.rolling(20).mean()
        rolling_vol = clean_returns.rolling(20).std()
        
        # Simple regime classification
        regime = pd.Series(index=clean_returns.index, dtype=str)
        regime[rolling_mean > rolling_vol] = 'bull'
        regime[rolling_mean < -rolling_vol] = 'bear'
        regime[(rolling_mean >= -rolling_vol) & (rolling_mean <= rolling_vol)] = 'neutral'
        
        result = pd.DataFrame({
            f'{self.name}_regime': regime,
            f'{self.name}_anomaly': (regime != regime.shift(1)).astype(int)
        }, index=clean_returns.index)
        
        return result.reindex(returns.index)


class WaveletAnalyzer:
    """
    Wavelet decomposition for multi-scale analysis.
    Detects patterns at different time scales.
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels
        self.name = "Wavelet"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Perform wavelet decomposition and detect multi-scale anomalies.
        """
        try:
            import pywt
            
            clean_series = series.dropna()
            
            if len(clean_series) < 2 ** self.levels:
                logger.warning("Insufficient data for wavelet analysis")
                return pd.DataFrame(index=series.index)
            
            # Wavelet decomposition
            coeffs = pywt.wavedec(clean_series.values, self.wavelet, level=self.levels)
            
            # Reconstruct at each level
            reconstructions = {}
            for i in range(self.levels + 1):
                coeffs_copy = [np.zeros_like(c) for c in coeffs]
                coeffs_copy[i] = coeffs[i]
                reconstructions[f'level_{i}'] = pywt.waverec(coeffs_copy, self.wavelet)[:len(clean_series)]
            
            # Energy at each level
            energies = [np.sum(c ** 2) for c in coeffs]
            total_energy = sum(energies)
            energy_ratios = [e / total_energy for e in energies]
            
            # Detect anomalies in detail coefficients
            detail_anomalies = []
            for i in range(1, len(coeffs)):
                detail = coeffs[i]
                zscore = (detail - detail.mean()) / (detail.std() + 1e-10)
                anomaly = np.abs(zscore) > 2.5
                detail_anomalies.append(anomaly)
            
            # Combine anomalies across scales
            # Upsample to match original length
            combined_anomaly = np.zeros(len(clean_series))
            for i, anom in enumerate(detail_anomalies):
                scale_factor = len(clean_series) // len(anom)
                upsampled = np.repeat(anom, scale_factor)[:len(clean_series)]
                combined_anomaly = combined_anomaly | upsampled
            
            result = pd.DataFrame({
                f'{self.name}_trend': reconstructions['level_0'],
                f'{self.name}_anomaly': combined_anomaly.astype(int),
                f'{self.name}_high_freq_energy': energy_ratios[-1] if energy_ratios else 0,
                f'{self.name}_score': combined_anomaly.astype(float)
            }, index=clean_series.index)
            
            return result.reindex(series.index)
            
        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet analysis")
            return pd.DataFrame(index=series.index)


class KalmanFilterAnalyzer:
    """
    Kalman Filter for optimal state estimation.
    Tracks true price level and detects deviations.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.name = "Kalman"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Apply Kalman filter and detect anomalies.
        """
        clean_series = series.dropna()
        
        # Initialize
        n = len(clean_series)
        
        # State estimate
        x_hat = np.zeros(n)
        x_hat[0] = clean_series.iloc[0]
        
        # Error covariance
        P = np.ones(n)
        P[0] = 1.0
        
        # Kalman gain
        K = np.zeros(n)
        
        # Innovation (prediction error)
        innovation = np.zeros(n)
        
        # Process noise covariance
        Q = self.process_noise
        
        # Measurement noise covariance
        R = self.measurement_noise
        
        for i in range(1, n):
            # Prediction
            x_pred = x_hat[i-1]
            P_pred = P[i-1] + Q
            
            # Update
            innovation[i] = clean_series.iloc[i] - x_pred
            S = P_pred + R  # Innovation covariance
            K[i] = P_pred / S
            
            x_hat[i] = x_pred + K[i] * innovation[i]
            P[i] = (1 - K[i]) * P_pred
        
        # Standardized innovation
        innovation_std = innovation / (np.sqrt(P + R) + 1e-10)
        
        # Anomaly: large innovations
        anomaly = np.abs(innovation_std) > 2.5
        
        result = pd.DataFrame({
            f'{self.name}_estimate': x_hat,
            f'{self.name}_innovation': innovation,
            f'{self.name}_innovation_std': innovation_std,
            f'{self.name}_kalman_gain': K,
            f'{self.name}_error_cov': P,
            f'{self.name}_anomaly': anomaly.astype(int),
            f'{self.name}_score': np.abs(innovation_std) / 2.5
        }, index=clean_series.index)
        
        return result.reindex(series.index)


class CyclicalModelEnsemble:
    """
    Ensemble of all cyclical and mean reversion models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzers = self._initialize_analyzers()
        
    def _initialize_analyzers(self) -> Dict:
        """Initialize all analyzers."""
        return {
            'fourier': FourierAnalyzer(
                n_harmonics=self.config.get('fourier_harmonics', 5)
            ),
            'hurst': HurstExponentAnalyzer(
                min_window=self.config.get('hurst_min_window', 20),
                max_window=self.config.get('hurst_max_window', 200)
            ),
            'ou': OrnsteinUhlenbeckAnalyzer(
                window=self.config.get('ou_window', 60)
            ),
            'hmm': HiddenMarkovModelAnalyzer(
                n_states=self.config.get('hmm_states', 3)
            ),
            'wavelet': WaveletAnalyzer(
                levels=self.config.get('wavelet_levels', 4)
            ),
            'kalman': KalmanFilterAnalyzer(
                process_noise=self.config.get('kalman_process_noise', 0.01),
                measurement_noise=self.config.get('kalman_measurement_noise', 0.1)
            )
        }
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cyclical analyzers to the data.
        """
        results = {}
        
        # Get price and returns series
        price = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = data['returns'] if 'returns' in data.columns else price.pct_change()
        
        logger.info("Running cyclical/mean reversion analysis...")
        
        # Fourier Analysis
        results['fourier'] = self.analyzers['fourier'].fit_transform(price)
        
        # Hurst Exponent
        results['hurst'] = self.analyzers['hurst'].fit_transform(returns)
        
        # Ornstein-Uhlenbeck
        results['ou'] = self.analyzers['ou'].fit_transform(price)
        
        # Hidden Markov Model
        results['hmm'] = self.analyzers['hmm'].fit_transform(returns)
        
        # Wavelet Analysis
        results['wavelet'] = self.analyzers['wavelet'].fit_transform(price)
        
        # Kalman Filter
        results['kalman'] = self.analyzers['kalman'].fit_transform(price)
        
        # Combine all results
        combined = pd.concat([df for df in results.values() if not df.empty], axis=1)
        
        # Calculate ensemble signals
        anomaly_cols = [col for col in combined.columns if col.endswith('_anomaly')]
        score_cols = [col for col in combined.columns if col.endswith('_score')]
        
        combined['cyclical_anomaly_count'] = combined[anomaly_cols].sum(axis=1)
        combined['cyclical_mean_score'] = combined[score_cols].mean(axis=1)
        combined['cyclical_ensemble_signal'] = (combined['cyclical_anomaly_count'] >= 2).astype(int)
        
        # Mean reversion signal
        if f'Hurst_mean_reverting' in combined.columns and f'OU_anomaly' in combined.columns:
            combined['mean_reversion_signal'] = (
                (combined.get('Hurst_mean_reverting', 0) == 1) & 
                (combined.get('OU_anomaly', 0) == 1)
            ).astype(int)
        
        logger.info(f"Cyclical ensemble complete: {len(combined.columns)} features generated")
        
        return combined


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Fetch RKLB data
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    data['returns'] = data['close'].pct_change()
    
    # Run ensemble
    ensemble = CyclicalModelEnsemble()
    results = ensemble.fit_transform(data)
    
    print("Cyclical Analysis Results:")
    print(results.tail(20))
    
    # Show anomaly summary
    anomaly_days = results[results['cyclical_ensemble_signal'] == 1]
    print(f"\nTotal cyclical ensemble signals: {len(anomaly_days)}")
