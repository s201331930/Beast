"""
Cyclical and Mean Reversion Analysis Module

Physics and Mathematics-inspired models for detecting cycles and mean reversion:
- Fourier Transform Analysis (frequency domain decomposition)
- Hilbert Transform (instantaneous phase and amplitude)
- Wavelet Transform (multi-resolution analysis)
- Hurst Exponent (long-range dependence)
- Ornstein-Uhlenbeck Process (mean reversion modeling)
- Half-Life of Mean Reversion
- Cointegration Analysis
- Regime Detection (Hidden Markov Models)
- Spectral Analysis
- Detrended Fluctuation Analysis (DFA)

Mathematical Foundation: Signal processing, stochastic processes, chaos theory
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class CyclicalAnalyzer:
    """
    Advanced cyclical and mean reversion analysis for financial data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price dataframe.
        
        Args:
            df: DataFrame with at least 'close' and 'returns' columns
        """
        self.df = df.copy()
        self.signals = pd.DataFrame(index=df.index)
        
    def fourier_analysis(self,
                         column: str = 'close',
                         detrend: bool = True,
                         max_harmonics: int = None) -> Dict:
        """
        Fourier Transform analysis for cycle detection.
        
        Mathematical basis:
        X(f) = ∫ x(t) * e^(-i2πft) dt
        
        Identifies dominant frequencies/cycles in the price series.
        
        Args:
            column: Column to analyze
            detrend: Whether to detrend before FFT
            max_harmonics: Maximum harmonics to extract
            
        Returns:
            Dictionary with frequency analysis results
        """
        max_harmonics = max_harmonics or config.cyclical.fourier_max_harmonics
        
        series = self.df[column].dropna()
        
        if detrend:
            # Remove linear trend
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series.values, 1)
            detrended = series.values - (slope * x + intercept)
        else:
            detrended = series.values
        
        # Apply FFT
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, 1)  # 1 day sampling
        
        # Get positive frequencies only
        positive_freq_idx = xf > 0
        freqs = xf[positive_freq_idx]
        amplitudes = 2.0/n * np.abs(yf[positive_freq_idx])
        phases = np.angle(yf[positive_freq_idx])
        
        # Convert to periods (trading days)
        periods = 1 / freqs
        
        # Find dominant cycles (top harmonics)
        top_idx = np.argsort(amplitudes)[-max_harmonics:][::-1]
        
        dominant_cycles = []
        for idx in top_idx:
            if periods[idx] > 2 and periods[idx] < len(series):  # Filter meaningful periods
                dominant_cycles.append({
                    'period': periods[idx],
                    'frequency': freqs[idx],
                    'amplitude': amplitudes[idx],
                    'phase': phases[idx],
                    'strength': amplitudes[idx] / np.sum(amplitudes)
                })
        
        # Reconstruct signal from dominant cycles
        reconstructed = np.zeros(n)
        for cycle in dominant_cycles[:5]:  # Top 5 cycles
            freq = cycle['frequency']
            amp = cycle['amplitude']
            phase = cycle['phase']
            t = np.arange(n)
            reconstructed += amp * np.cos(2 * np.pi * freq * t + phase)
        
        self.signals['fourier_reconstructed'] = pd.Series(reconstructed, index=series.index)
        self.signals['fourier_residual'] = series.values - reconstructed
        
        # Cycle position indicator (where are we in the dominant cycle)
        if dominant_cycles:
            dom_period = dominant_cycles[0]['period']
            dom_phase = dominant_cycles[0]['phase']
            cycle_position = np.mod(np.arange(n) / dom_period + dom_phase / (2 * np.pi), 1)
            self.signals['cycle_position'] = pd.Series(cycle_position, index=series.index)
        
        return {
            'dominant_cycles': dominant_cycles,
            'frequencies': freqs,
            'amplitudes': amplitudes,
            'periods': periods,
            'reconstructed': reconstructed
        }
    
    def hilbert_transform(self, column: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """
        Hilbert Transform for instantaneous phase and amplitude.
        
        Mathematical basis:
        Analytic signal: z(t) = x(t) + i*H[x(t)]
        Amplitude: A(t) = |z(t)|
        Phase: φ(t) = arg(z(t))
        
        Useful for detecting cycle phase transitions.
        
        Args:
            column: Column to analyze
            
        Returns:
            Tuple of (amplitude envelope, instantaneous phase)
        """
        series = self.df[column].dropna()
        
        # Detrend
        detrended = signal.detrend(series.values)
        
        # Hilbert transform
        analytic_signal = signal.hilbert(detrended)
        
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi)
        
        self.signals['hilbert_amplitude'] = pd.Series(amplitude_envelope, index=series.index)
        self.signals['hilbert_phase'] = pd.Series(instantaneous_phase, index=series.index)
        self.signals['hilbert_frequency'] = pd.Series(
            np.concatenate([[np.nan], instantaneous_frequency]),
            index=series.index
        )
        
        # Phase-based signals
        phase_mod = np.mod(instantaneous_phase, 2 * np.pi)
        self.signals['cycle_phase_normalized'] = pd.Series(phase_mod / (2 * np.pi), index=series.index)
        
        # Detect phase reversals (potential turning points)
        phase_velocity = np.diff(phase_mod)
        phase_reversal = np.abs(phase_velocity) > np.pi
        self.signals['phase_reversal'] = pd.Series(
            np.concatenate([[False], phase_reversal]),
            index=series.index
        )
        
        return (
            pd.Series(amplitude_envelope, index=series.index),
            pd.Series(instantaneous_phase, index=series.index)
        )
    
    def wavelet_analysis(self,
                         column: str = 'close',
                         wavelet: str = 'morl',
                         scales: np.ndarray = None) -> Dict:
        """
        Continuous Wavelet Transform for multi-resolution analysis.
        
        Mathematical basis:
        W(a,b) = (1/√a) ∫ x(t) * ψ*((t-b)/a) dt
        
        Provides time-frequency localization.
        
        Args:
            column: Column to analyze
            wavelet: Wavelet type ('morl', 'cgau', 'mexh')
            scales: Scales to analyze
            
        Returns:
            Dictionary with wavelet coefficients and analysis
        """
        series = self.df[column].dropna()
        
        if scales is None:
            # Scales corresponding to ~5 to ~252 day periods
            scales = np.arange(5, min(100, len(series)//4))
        
        # Detrend
        detrended = signal.detrend(series.values)
        
        # Continuous wavelet transform using scipy
        from scipy.signal import cwt, morlet2
        
        # Use morlet wavelet
        widths = scales
        cwtmatr = cwt(detrended, morlet2, widths)
        
        # Power spectrum
        power = np.abs(cwtmatr) ** 2
        
        # Find dominant scale at each time point
        dominant_scale = scales[np.argmax(power, axis=0)]
        
        self.signals['wavelet_dominant_scale'] = pd.Series(dominant_scale, index=series.index)
        
        # Average power at each scale (global wavelet spectrum)
        global_power = np.mean(power, axis=1)
        
        # Scale-averaged power (energy in specific frequency bands)
        # Short-term (5-20 days)
        short_term_idx = (scales >= 5) & (scales <= 20)
        if np.any(short_term_idx):
            self.signals['wavelet_short_power'] = pd.Series(
                np.mean(power[short_term_idx, :], axis=0),
                index=series.index
            )
        
        # Medium-term (20-60 days)
        medium_term_idx = (scales >= 20) & (scales <= 60)
        if np.any(medium_term_idx):
            self.signals['wavelet_medium_power'] = pd.Series(
                np.mean(power[medium_term_idx, :], axis=0),
                index=series.index
            )
        
        return {
            'coefficients': cwtmatr,
            'power': power,
            'scales': scales,
            'global_power': global_power,
            'dominant_scale': dominant_scale
        }
    
    def hurst_exponent(self,
                       column: str = 'close',
                       min_window: int = None,
                       max_window: int = None) -> float:
        """
        Calculate Hurst Exponent using R/S analysis.
        
        Mathematical basis:
        E[R(n)/S(n)] = C * n^H
        
        H < 0.5: Mean reverting (anti-persistent)
        H = 0.5: Random walk
        H > 0.5: Trending (persistent)
        
        Args:
            column: Column to analyze
            min_window: Minimum window size
            max_window: Maximum window size
            
        Returns:
            Hurst exponent value
        """
        min_window = min_window or config.cyclical.hurst_min_window
        max_window = max_window or config.cyclical.hurst_max_window
        
        series = self.df[column].dropna().values
        n = len(series)
        
        if n < max_window:
            max_window = n // 2
        
        # R/S analysis
        rs_list = []
        n_list = []
        
        for window in range(min_window, max_window + 1, 5):
            rs_values = []
            
            for start in range(0, n - window, window // 2):
                subset = series[start:start + window]
                
                # Mean-adjusted cumulative sum
                mean_adj = subset - np.mean(subset)
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(subset, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                rs_list.append(np.mean(rs_values))
                n_list.append(window)
        
        # Log-log regression
        if len(rs_list) > 2:
            log_n = np.log(n_list)
            log_rs = np.log(rs_list)
            
            slope, intercept = np.polyfit(log_n, log_rs, 1)
            hurst = slope
        else:
            hurst = 0.5
        
        self.signals['hurst_exponent'] = hurst
        
        # Rolling Hurst
        rolling_hurst = []
        window_size = 100
        
        for i in range(window_size, len(series)):
            subset = series[i-window_size:i]
            h = self._calculate_hurst_rs(subset, min_window, min(50, window_size//2))
            rolling_hurst.append(h)
        
        rolling_hurst = [np.nan] * window_size + rolling_hurst
        self.signals['rolling_hurst'] = pd.Series(rolling_hurst, index=self.df[column].dropna().index[:len(rolling_hurst)])
        
        return hurst
    
    def _calculate_hurst_rs(self, series: np.ndarray, min_w: int, max_w: int) -> float:
        """Helper for rolling Hurst calculation."""
        n = len(series)
        if n < max_w:
            return 0.5
        
        rs_list = []
        n_list = []
        
        for window in range(min_w, max_w + 1, 3):
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
            slope, _ = np.polyfit(np.log(n_list), np.log(rs_list), 1)
            return slope
        return 0.5
    
    def ornstein_uhlenbeck(self, column: str = 'close') -> Dict:
        """
        Fit Ornstein-Uhlenbeck process for mean reversion analysis.
        
        Mathematical basis:
        dx = θ(μ - x)dt + σdW
        
        θ: Speed of mean reversion
        μ: Long-term mean
        σ: Volatility
        
        Half-life = ln(2) / θ
        
        Args:
            column: Column to analyze
            
        Returns:
            Dictionary with OU parameters
        """
        series = self.df[column].dropna()
        log_price = np.log(series.values)
        
        # Estimate parameters using AR(1)
        # x_{t+1} = a + b*x_t + ε
        # θ = -ln(b), μ = a/(1-b), σ = std(ε)/sqrt(dt)
        
        x = log_price[:-1]
        y = log_price[1:]
        
        # OLS regression
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        a, b = beta
        residuals = y - (a + b * x)
        sigma_residual = np.std(residuals)
        
        if b > 0 and b < 1:
            theta = -np.log(b)  # Daily
            mu = a / (1 - b)
            sigma = sigma_residual * np.sqrt(2 * theta)
            half_life = np.log(2) / theta
        else:
            # Non-mean-reverting
            theta = 0
            mu = np.mean(log_price)
            sigma = sigma_residual
            half_life = np.inf
        
        # Store results
        ou_params = {
            'theta': theta,
            'mu': mu,
            'mu_price': np.exp(mu),
            'sigma': sigma,
            'half_life': half_life,
            'b_coefficient': b
        }
        
        # Current deviation from mean
        deviation = log_price - mu
        self.signals['ou_deviation'] = pd.Series(deviation, index=series.index)
        self.signals['ou_deviation_pct'] = self.signals['ou_deviation'] * 100
        
        # Expected reversion (where price should be in half_life days)
        expected_reversion = mu + deviation * np.exp(-theta * min(half_life, 30))
        self.signals['ou_expected_price'] = pd.Series(np.exp(expected_reversion), index=series.index)
        
        # Mean reversion signal (-1 to 1)
        std_dev = np.std(deviation)
        mr_signal = -deviation / (2 * std_dev)  # Negative deviation = bullish
        mr_signal = np.clip(mr_signal, -1, 1)
        self.signals['ou_mr_signal'] = pd.Series(mr_signal, index=series.index)
        
        return ou_params
    
    def detrended_fluctuation_analysis(self,
                                       column: str = 'close',
                                       min_box: int = 4,
                                       max_box: int = None) -> float:
        """
        Detrended Fluctuation Analysis (DFA) for long-range correlations.
        
        Mathematical basis:
        F(n) ~ n^α
        
        α < 0.5: Anti-correlated
        α = 0.5: Uncorrelated (white noise)
        α > 0.5: Long-range correlated
        α = 1.0: 1/f noise
        α > 1.0: Non-stationary
        
        Args:
            column: Column to analyze
            min_box: Minimum box size
            max_box: Maximum box size
            
        Returns:
            DFA scaling exponent (alpha)
        """
        series = self.df[column].dropna()
        data = series.values
        n = len(data)
        
        if max_box is None:
            max_box = n // 4
        
        # Integrated series
        y = np.cumsum(data - np.mean(data))
        
        # Box sizes (logarithmically spaced)
        box_sizes = np.unique(np.logspace(
            np.log10(min_box),
            np.log10(max_box),
            50
        ).astype(int))
        
        fluctuations = []
        
        for box_size in box_sizes:
            n_boxes = n // box_size
            
            if n_boxes < 2:
                continue
            
            f2 = []
            
            for i in range(n_boxes):
                start = i * box_size
                end = start + box_size
                segment = y[start:end]
                
                # Fit linear trend
                x = np.arange(box_size)
                slope, intercept = np.polyfit(x, segment, 1)
                trend = slope * x + intercept
                
                # Fluctuation (RMS of residuals)
                f2.append(np.mean((segment - trend) ** 2))
            
            fluctuations.append(np.sqrt(np.mean(f2)))
        
        # Log-log regression
        if len(fluctuations) > 2:
            valid_sizes = box_sizes[:len(fluctuations)]
            log_n = np.log(valid_sizes)
            log_f = np.log(fluctuations)
            
            alpha, intercept = np.polyfit(log_n, log_f, 1)
        else:
            alpha = 0.5
        
        self.signals['dfa_alpha'] = alpha
        
        return alpha
    
    def spectral_entropy(self, column: str = 'returns', window: int = 50) -> pd.Series:
        """
        Calculate spectral entropy for market complexity/regime detection.
        
        Mathematical basis:
        H = -Σ P(f) * log(P(f))
        
        Where P(f) is normalized power spectral density.
        High entropy = complex/unpredictable, Low entropy = regular/predictable
        
        Args:
            column: Column to analyze
            window: Rolling window
            
        Returns:
            Series of spectral entropy values
        """
        series = self.df[column].dropna()
        
        entropy = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            segment = series.iloc[i-window:i].values
            
            # Power spectral density
            freqs, psd = signal.welch(segment, fs=1.0, nperseg=min(window//2, 32))
            
            # Normalize to probability distribution
            psd_norm = psd / np.sum(psd)
            
            # Shannon entropy
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
            H = -np.sum(psd_norm * np.log2(psd_norm))
            
            # Normalize by maximum entropy
            H_max = np.log2(len(freqs))
            entropy.iloc[i] = H / H_max if H_max > 0 else 0
        
        self.signals['spectral_entropy'] = entropy
        
        # Low entropy suggests regime/pattern
        self.signals['low_entropy_regime'] = entropy < 0.7
        
        return entropy
    
    def regime_detection_hmm(self, n_states: int = None) -> pd.Series:
        """
        Hidden Markov Model for regime detection.
        
        Identifies different market regimes (e.g., trending, ranging, volatile).
        
        Args:
            n_states: Number of hidden states
            
        Returns:
            Series of regime labels
        """
        n_states = n_states or config.cyclical.regime_states
        
        returns = self.df['returns'].dropna()
        
        try:
            from hmmlearn import hmm
            
            # Prepare features
            features = np.column_stack([
                returns.values.reshape(-1, 1),
                self.df['volatility_10'].loc[returns.index].fillna(method='bfill').values.reshape(-1, 1)
                if 'volatility_10' in self.df.columns else np.zeros((len(returns), 1))
            ])
            
            # Remove NaNs
            valid_idx = ~np.isnan(features).any(axis=1)
            features_clean = features[valid_idx]
            
            # Fit HMM
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=1000,
                random_state=42
            )
            model.fit(features_clean)
            
            # Predict regimes
            regimes = np.full(len(returns), np.nan)
            regimes[valid_idx] = model.predict(features_clean)
            
            self.signals['hmm_regime'] = pd.Series(regimes, index=returns.index)
            
            # Regime transition probabilities
            trans_probs = model.transmat_
            
            # Regime means
            regime_means = model.means_
            
            return pd.Series(regimes, index=returns.index)
            
        except ImportError:
            print("hmmlearn not available, using simple regime detection")
            
            # Fallback: simple volatility-based regime
            vol = self.df['volatility_20'] if 'volatility_20' in self.df.columns else returns.rolling(20).std()
            vol_percentile = vol.rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) > 20 else 0.5
            )
            
            regimes = pd.cut(vol_percentile, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2])
            self.signals['hmm_regime'] = regimes.astype(float)
            
            return regimes.astype(float)
    
    def bollinger_signals(self,
                          window: int = None,
                          num_std: float = None) -> pd.DataFrame:
        """
        Enhanced Bollinger Band signals for mean reversion.
        
        Args:
            window: Rolling window
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger signals
        """
        window = window or config.technical.bb_window
        num_std = num_std or config.technical.bb_std
        
        close = self.df['close']
        
        # Standard Bollinger Bands
        middle = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        # %B indicator (position within bands)
        pct_b = (close - lower) / (upper - lower)
        
        # Bandwidth (volatility indicator)
        bandwidth = (upper - lower) / middle
        
        # Signals
        self.signals['bb_pct_b'] = pct_b
        self.signals['bb_bandwidth'] = bandwidth
        
        # Squeeze detection (low volatility)
        bw_percentile = bandwidth.rolling(126).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) > 20 else 0.5
        )
        self.signals['bb_squeeze'] = bw_percentile < 0.2
        
        # Mean reversion signal
        self.signals['bb_mr_signal'] = -2 * (pct_b - 0.5)  # -1 to 1, negative when overbought
        
        # Extreme readings
        self.signals['bb_oversold'] = pct_b < 0
        self.signals['bb_overbought'] = pct_b > 1
        
        return self.signals[['bb_pct_b', 'bb_bandwidth', 'bb_squeeze', 'bb_mr_signal']]
    
    def rsi_divergence(self, window: int = None) -> pd.DataFrame:
        """
        RSI divergence analysis for potential reversals.
        
        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        
        Args:
            window: RSI window
            
        Returns:
            DataFrame with RSI divergence signals
        """
        window = window or config.technical.rsi_window
        
        close = self.df['close']
        
        # Calculate RSI if not present
        if 'rsi' not in self.df.columns:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = self.df['rsi']
        
        # Find local minima and maxima
        lookback = 5
        
        # Price extremes
        price_min = close.rolling(lookback * 2 + 1, center=True).min() == close
        price_max = close.rolling(lookback * 2 + 1, center=True).max() == close
        
        # RSI extremes
        rsi_min = rsi.rolling(lookback * 2 + 1, center=True).min() == rsi
        rsi_max = rsi.rolling(lookback * 2 + 1, center=True).max() == rsi
        
        # Divergence detection (simplified)
        bullish_div = pd.Series(False, index=close.index)
        bearish_div = pd.Series(False, index=close.index)
        
        for i in range(lookback * 4, len(close)):
            # Look for bullish divergence
            if price_min.iloc[i]:
                # Find previous price low
                prev_lows = close.iloc[i-lookback*4:i-lookback][price_min.iloc[i-lookback*4:i-lookback]]
                if len(prev_lows) > 0:
                    if close.iloc[i] < prev_lows.iloc[-1]:  # Lower price low
                        # Check if RSI made higher low
                        prev_rsi = rsi.iloc[prev_lows.index[-1]]
                        curr_rsi = rsi.iloc[i]
                        if curr_rsi > prev_rsi:
                            bullish_div.iloc[i] = True
            
            # Look for bearish divergence
            if price_max.iloc[i]:
                prev_highs = close.iloc[i-lookback*4:i-lookback][price_max.iloc[i-lookback*4:i-lookback]]
                if len(prev_highs) > 0:
                    if close.iloc[i] > prev_highs.iloc[-1]:  # Higher price high
                        prev_rsi = rsi.iloc[prev_highs.index[-1]]
                        curr_rsi = rsi.iloc[i]
                        if curr_rsi < prev_rsi:
                            bearish_div.iloc[i] = True
        
        self.signals['rsi'] = rsi
        self.signals['rsi_bullish_div'] = bullish_div
        self.signals['rsi_bearish_div'] = bearish_div
        
        # RSI extreme signals
        self.signals['rsi_oversold'] = rsi < config.technical.rsi_oversold
        self.signals['rsi_overbought'] = rsi > config.technical.rsi_overbought
        
        return self.signals[['rsi', 'rsi_bullish_div', 'rsi_bearish_div']]
    
    def run_all_analysis(self) -> pd.DataFrame:
        """
        Run all cyclical and mean reversion analysis.
        
        Returns:
            DataFrame with all signals
        """
        print("=" * 60)
        print("Running Cyclical and Mean Reversion Analysis")
        print("=" * 60)
        
        # Fourier analysis
        print("Running Fourier analysis...")
        fourier_results = self.fourier_analysis()
        print(f"  Found {len(fourier_results['dominant_cycles'])} dominant cycles")
        
        # Hilbert transform
        print("Running Hilbert transform...")
        self.hilbert_transform()
        
        # Wavelet analysis
        print("Running Wavelet analysis...")
        self.wavelet_analysis()
        
        # Hurst exponent
        print("Calculating Hurst exponent...")
        hurst = self.hurst_exponent()
        print(f"  Hurst exponent: {hurst:.3f}")
        if hurst < 0.5:
            print("  → Mean reverting behavior detected")
        elif hurst > 0.5:
            print("  → Trending behavior detected")
        
        # Ornstein-Uhlenbeck
        print("Fitting Ornstein-Uhlenbeck process...")
        ou_params = self.ornstein_uhlenbeck()
        print(f"  Half-life: {ou_params['half_life']:.1f} days")
        print(f"  Mean price: ${ou_params['mu_price']:.2f}")
        
        # DFA
        print("Running Detrended Fluctuation Analysis...")
        dfa_alpha = self.detrended_fluctuation_analysis()
        print(f"  DFA alpha: {dfa_alpha:.3f}")
        
        # Spectral entropy
        print("Calculating spectral entropy...")
        self.spectral_entropy()
        
        # Regime detection
        print("Detecting market regimes...")
        self.regime_detection_hmm()
        
        # Bollinger signals
        print("Calculating Bollinger Band signals...")
        self.bollinger_signals()
        
        # RSI divergence
        print("Analyzing RSI divergence...")
        self.rsi_divergence()
        
        print("=" * 60)
        print("Cyclical Analysis Complete")
        print(f"Generated {len(self.signals.columns)} signal columns")
        print("=" * 60)
        
        return self.signals
    
    def get_cycle_summary(self) -> Dict:
        """Get summary of cyclical analysis."""
        summary = {
            'hurst_exponent': self.signals.get('hurst_exponent', None),
            'dfa_alpha': self.signals.get('dfa_alpha', None),
            'current_regime': self.signals['hmm_regime'].iloc[-1] if 'hmm_regime' in self.signals.columns else None,
            'current_bb_position': self.signals['bb_pct_b'].iloc[-1] if 'bb_pct_b' in self.signals.columns else None,
            'current_rsi': self.signals['rsi'].iloc[-1] if 'rsi' in self.signals.columns else None,
            'ou_deviation': self.signals['ou_deviation'].iloc[-1] if 'ou_deviation' in self.signals.columns else None,
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
    
    # Run cyclical analysis
    analyzer = CyclicalAnalyzer(df)
    signals = analyzer.run_all_analysis()
    
    # Print summary
    summary = analyzer.get_cycle_summary()
    print("\nCyclical Analysis Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
