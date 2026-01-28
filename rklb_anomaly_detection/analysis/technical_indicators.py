"""
Technical Indicators Module
===========================
Comprehensive technical analysis indicators for signal generation:
- Momentum indicators (RSI, MACD, Stochastic)
- Volume indicators (OBV, MFI, VWAP, Accumulation/Distribution)
- Trend indicators (ADX, Ichimoku, Supertrend)
- Volatility indicators (ATR, Keltner Channels)
- Custom composite indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger


class TechnicalIndicators:
    """
    Calculates comprehensive technical indicators for anomaly detection.
    """
    
    def __init__(self):
        self.name = "TechnicalIndicators"
        
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Expected columns: open, high, low, close, volume
        """
        df = data.copy()
        
        # Ensure lowercase column names
        df.columns = [c.lower() for c in df.columns]
        
        # Price-based calculations
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            # Momentum Indicators
            df = self._calculate_rsi(df)
            df = self._calculate_macd(df)
            df = self._calculate_stochastic(df)
            df = self._calculate_cci(df)
            df = self._calculate_williams_r(df)
            df = self._calculate_roc(df)
            
            # Trend Indicators
            df = self._calculate_adx(df)
            df = self._calculate_aroon(df)
            df = self._calculate_ichimoku(df)
            df = self._calculate_supertrend(df)
            
            # Volatility Indicators
            df = self._calculate_atr(df)
            df = self._calculate_keltner_channels(df)
            df = self._calculate_donchian_channels(df)
            
            # Support/Resistance
            df = self._calculate_pivot_points(df)
        
        # Volume Indicators
        if 'volume' in df.columns:
            df = self._calculate_obv(df)
            df = self._calculate_mfi(df)
            df = self._calculate_vwap(df)
            df = self._calculate_accumulation_distribution(df)
            df = self._calculate_force_index(df)
            df = self._calculate_volume_profile(df)
        
        # Composite Signals
        df = self._calculate_composite_signals(df)
        
        logger.info(f"Calculated {len([c for c in df.columns if c not in data.columns])} technical indicators")
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_divergence'] = self._detect_divergence(df['close'], df['RSI'])
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # MACD signals
        df['MACD_bullish_cross'] = ((df['MACD'] > df['MACD_signal']) & 
                                    (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
        df['MACD_bearish_cross'] = ((df['MACD'] < df['MACD_signal']) & 
                                    (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))).astype(int)
        df['MACD_divergence'] = self._detect_divergence(df['close'], df['MACD'])
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator."""
        lowest_low = df['low'].rolling(window=k_window).min()
        highest_high = df['high'].rolling(window=k_window).max()
        
        df['Stoch_K'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_window).mean()
        
        # Stochastic signals
        df['Stoch_overbought'] = (df['Stoch_K'] > 80).astype(int)
        df['Stoch_oversold'] = (df['Stoch_K'] < 20).astype(int)
        df['Stoch_bullish_cross'] = ((df['Stoch_K'] > df['Stoch_D']) & 
                                     (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) &
                                     (df['Stoch_K'] < 20)).astype(int)
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        df['CCI'] = (typical_price - sma) / (0.015 * mad + 1e-10)
        df['CCI_extreme'] = (np.abs(df['CCI']) > 100).astype(int)
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Williams %R."""
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        
        df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
        df['Williams_R_overbought'] = (df['Williams_R'] > -20).astype(int)
        df['Williams_R_oversold'] = (df['Williams_R'] < -80).astype(int)
        
        return df
    
    def _calculate_roc(self, df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        """Rate of Change."""
        df['ROC'] = ((df['close'] - df['close'].shift(window)) / df['close'].shift(window)) * 100
        df['ROC_acceleration'] = df['ROC'].diff()
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Average Directional Index."""
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / (atr + 1e-10)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(window=window).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # ADX signals
        df['ADX_strong_trend'] = (df['ADX'] > 25).astype(int)
        df['ADX_bullish'] = ((df['Plus_DI'] > df['Minus_DI']) & (df['ADX'] > 25)).astype(int)
        
        return df
    
    def _calculate_aroon(self, df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
        """Aroon Indicator."""
        def aroon_up(close):
            return 100 * (window - close.rolling(window + 1).apply(lambda x: window - x.argmax(), raw=True)) / window
        
        def aroon_down(close):
            return 100 * (window - close.rolling(window + 1).apply(lambda x: window - x.argmin(), raw=True)) / window
        
        df['Aroon_Up'] = aroon_up(df['high'])
        df['Aroon_Down'] = aroon_down(df['low'])
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
        
        return df
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud."""
        # Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
        
        # Signals
        df['Ichimoku_above_cloud'] = ((df['close'] > df['Ichimoku_SpanA']) & 
                                      (df['close'] > df['Ichimoku_SpanB'])).astype(int)
        df['Ichimoku_bullish_cross'] = ((df['Ichimoku_Tenkan'] > df['Ichimoku_Kijun']) &
                                        (df['Ichimoku_Tenkan'].shift(1) <= df['Ichimoku_Kijun'].shift(1))).astype(int)
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Supertrend Indicator."""
        atr = self._atr(df, period)
        
        hl2 = (df['high'] + df['low']) / 2
        
        # Upper and Lower Bands
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        
        df['Supertrend'] = supertrend
        df['Supertrend_direction'] = direction
        df['Supertrend_signal'] = (direction != direction.shift(1)).astype(int)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Average True Range."""
        tr = self._true_range(df)
        df['ATR'] = tr.rolling(window=window).mean()
        df['ATR_percent'] = df['ATR'] / df['close'] * 100
        df['ATR_expansion'] = (df['ATR'] > df['ATR'].rolling(20).mean() * 1.5).astype(int)
        
        return df
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
        """Keltner Channels."""
        ema = df['close'].ewm(span=window, adjust=False).mean()
        atr = self._atr(df, window)
        
        df['Keltner_upper'] = ema + atr_mult * atr
        df['Keltner_lower'] = ema - atr_mult * atr
        df['Keltner_mid'] = ema
        
        df['Keltner_breakout_up'] = (df['close'] > df['Keltner_upper']).astype(int)
        df['Keltner_breakout_down'] = (df['close'] < df['Keltner_lower']).astype(int)
        
        return df
    
    def _calculate_donchian_channels(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Donchian Channels."""
        df['Donchian_upper'] = df['high'].rolling(window=window).max()
        df['Donchian_lower'] = df['low'].rolling(window=window).min()
        df['Donchian_mid'] = (df['Donchian_upper'] + df['Donchian_lower']) / 2
        df['Donchian_width'] = (df['Donchian_upper'] - df['Donchian_lower']) / df['Donchian_mid']
        
        df['Donchian_breakout_up'] = (df['close'] == df['Donchian_upper']).astype(int)
        df['Donchian_breakout_down'] = (df['close'] == df['Donchian_lower']).astype(int)
        
        return df
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot Points (daily levels)."""
        df['Pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['Pivot_R1'] = 2 * df['Pivot'] - df['low'].shift(1)
        df['Pivot_S1'] = 2 * df['Pivot'] - df['high'].shift(1)
        df['Pivot_R2'] = df['Pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['Pivot_S2'] = df['Pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume."""
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                       np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['OBV'] = np.cumsum(obv)
        df['OBV_ma20'] = df['OBV'].rolling(20).mean()
        df['OBV_divergence'] = self._detect_divergence(df['close'], df['OBV'])
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_mf / (negative_mf + 1e-10)
        df['MFI'] = 100 - (100 / (1 + money_ratio))
        
        df['MFI_overbought'] = (df['MFI'] > 80).astype(int)
        df['MFI_oversold'] = (df['MFI'] < 20).astype(int)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price (rolling)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['VWAP_distance'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
        
        return df
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accumulation/Distribution Line."""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['AD'] = (clv * df['volume']).cumsum()
        df['AD_ma20'] = df['AD'].rolling(20).mean()
        
        return df
    
    def _calculate_force_index(self, df: pd.DataFrame, window: int = 13) -> pd.DataFrame:
        """Force Index."""
        df['Force_Index'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df['Force_Index_EMA'] = df['Force_Index'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def _calculate_volume_profile(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Volume profile indicators."""
        df['Volume_SMA'] = df['volume'].rolling(window=window).mean()
        df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
        df['Volume_spike'] = (df['Volume_ratio'] > 2.0).astype(int)
        
        # Price-Volume correlation
        df['PV_correlation'] = df['close'].pct_change().rolling(window).corr(df['volume'].pct_change())
        
        return df
    
    def _calculate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite momentum and trend signals."""
        # Composite Momentum Score (normalize and combine)
        momentum_indicators = ['RSI', 'Stoch_K', 'CCI', 'MFI']
        available_momentum = [col for col in momentum_indicators if col in df.columns]
        
        if available_momentum:
            # Normalize each indicator to 0-100 scale
            momentum_normalized = df[available_momentum].copy()
            if 'CCI' in momentum_normalized.columns:
                momentum_normalized['CCI'] = (momentum_normalized['CCI'] + 300) / 6 * 100  # Normalize CCI
            
            df['Composite_Momentum'] = momentum_normalized.mean(axis=1)
            df['Momentum_extreme_bullish'] = (df['Composite_Momentum'] > 70).astype(int)
            df['Momentum_extreme_bearish'] = (df['Composite_Momentum'] < 30).astype(int)
        
        # Trend Strength
        trend_signals = ['ADX_strong_trend', 'Ichimoku_above_cloud', 'Supertrend_direction']
        available_trend = [col for col in trend_signals if col in df.columns]
        
        if available_trend:
            df['Trend_alignment'] = df[available_trend].mean(axis=1)
        
        # Volume Confirmation
        volume_signals = ['Volume_spike', 'OBV_divergence']
        available_volume = [col for col in volume_signals if col in df.columns]
        
        if available_volume:
            df['Volume_confirmation'] = df[available_volume].max(axis=1)
        
        return df
    
    def _detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> pd.Series:
        """Detect bullish/bearish divergence."""
        # Simplified divergence detection
        price_change = price.pct_change(window)
        indicator_change = indicator.pct_change(window)
        
        # Bullish divergence: price lower, indicator higher
        bullish_div = (price_change < -0.05) & (indicator_change > 0.05)
        
        # Bearish divergence: price higher, indicator lower
        bearish_div = (price_change > 0.05) & (indicator_change < -0.05)
        
        divergence = bullish_div.astype(int) - bearish_div.astype(int)
        return divergence
    
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr = self._true_range(df)
        return tr.rolling(window=window).mean()


if __name__ == "__main__":
    import yfinance as yf
    
    # Test with RKLB data
    data = yf.Ticker("RKLB").history(period="2y")
    
    ti = TechnicalIndicators()
    result = ti.calculate_all(data)
    
    print("Technical Indicators calculated:")
    print(result.tail(10))
    print(f"\nTotal columns: {len(result.columns)}")
