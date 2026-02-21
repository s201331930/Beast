#!/usr/bin/env python3
"""
CHAMPIONSHIP TRADING SCANNER
============================
Combines scientific screening with proven technical strategies used by
championship traders (Mark Minervini, William O'Neil, Stan Weinstein).

KEY STRATEGIES IMPLEMENTED:
1. Volatility Contraction Pattern (VCP) - Minervini
2. Relative Strength + Base Breakout - O'Neil's CANSLIM
3. Accumulation/Distribution - Smart Money Tracking
4. Stage Analysis - Weinstein's Method
5. Momentum + Mean Reversion Hybrid

TIME SERIES FLAGS:
- Tracks consolidation days, accumulation score, RS ranking over time
- Identifies stocks coiling for potential explosive moves
- Combines multiple timeframes for confluence

MARKET REGIME FILTERS:
- Overall market trend (bull/bear/neutral)
- Volatility regime (expansion/contraction)
- Market breadth (participation)
- Sector rotation status
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


# =============================================================================
# TASI TICKERS
# =============================================================================

TASI_TICKERS = [
    '1180.SR', '1010.SR', '1150.SR', '1140.SR', '1060.SR', '1050.SR',
    '1020.SR', '1030.SR', '1080.SR', '2222.SR', '2030.SR', '4030.SR',
    '2381.SR', '2380.SR', '2010.SR', '2290.SR', '2250.SR', '2210.SR',
    '2001.SR', '2060.SR', '2310.SR', '2350.SR', '2330.SR', '2170.SR',
    '2020.SR', '2190.SR', '1211.SR', '1320.SR', '1321.SR', '1302.SR',
    '1304.SR', '2200.SR', '2220.SR', '2240.SR', '2320.SR', '2370.SR',
    '3010.SR', '3020.SR', '3030.SR', '3040.SR', '3050.SR', '3060.SR',
    '3080.SR', '3090.SR', '5110.SR', '2082.SR', '2083.SR', '4190.SR',
    '4003.SR', '4240.SR', '4001.SR', '4002.SR', '4004.SR', '4007.SR',
    '4009.SR', '4020.SR', '4031.SR', '4050.SR', '4080.SR', '4110.SR',
    '4140.SR', '4200.SR', '4220.SR', '4250.SR', '4270.SR', '4280.SR',
    '4290.SR', '2280.SR', '2050.SR', '6002.SR', '6001.SR', '6010.SR',
    '4071.SR', '7010.SR', '7020.SR', '7030.SR', '4300.SR', '4310.SR',
    '4320.SR', '4330.SR', '4331.SR', '4332.SR', '4333.SR', '1120.SR',
    '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR', '8050.SR',
    '8060.SR', '8100.SR', '8120.SR', '8150.SR', '8160.SR', '8180.SR',
    '8200.SR', '8210.SR', '8230.SR', '8240.SR', '8250.SR', '8300.SR',
    '8310.SR', '1111.SR',
]

TASI_TICKERS = list(set(TASI_TICKERS))


# =============================================================================
# TECHNICAL FLAGS DATA STRUCTURES
# =============================================================================

@dataclass
class TechnicalFlags:
    """All technical flags for a stock."""
    ticker: str
    name: str
    
    # === PRICE ACTION FLAGS ===
    consolidation_days: int          # Days price in tight range
    consolidation_tightness: float   # How tight the range is (ATR ratio)
    breakout_proximity: float        # % away from breakout level
    stage: str                       # Stage 1/2/3/4 (Weinstein)
    
    # === VOLATILITY FLAGS ===
    vcp_score: float                 # Volatility Contraction Pattern score
    bb_squeeze: bool                 # Bollinger Band squeeze active
    bb_squeeze_days: int             # Days in squeeze
    atr_contraction: float           # ATR vs 20-day avg ATR
    
    # === MOMENTUM FLAGS ===
    rs_rank: int                     # Relative Strength rank (1-100)
    rs_new_high: bool                # RS at new high
    momentum_20d: float              # 20-day momentum
    momentum_60d: float              # 60-day momentum
    momentum_accel: float            # Momentum acceleration
    
    # === VOLUME FLAGS ===
    accumulation_score: float        # Accumulation/Distribution score
    volume_trend: str                # ACCUMULATING/DISTRIBUTING/NEUTRAL
    obv_divergence: str              # BULLISH_DIV/BEARISH_DIV/NONE
    volume_dry_up: bool              # Volume contracting (base building)
    
    # === TREND FLAGS ===
    above_20ma: bool
    above_50ma: bool
    above_200ma: bool
    ma_alignment: str                # BULLISH/BEARISH/MIXED
    trend_strength: float            # ADX value
    
    # === PATTERN FLAGS ===
    base_pattern: str                # CUP/FLAG/FLAT_BASE/VCP/NONE
    base_depth: float                # Depth of base (%)
    base_length_weeks: int           # Length of base
    handle_forming: bool             # Handle forming in cup
    
    # === COMPOSITE SCORES ===
    setup_quality: float             # Overall setup quality (0-100)
    timing_score: float              # How good is timing now (0-100)
    risk_reward: float               # Estimated risk/reward ratio


@dataclass
class MarketRegime:
    """Market-wide regime indicators."""
    trend: str                       # BULL/BEAR/NEUTRAL
    trend_strength: float            # 0-100
    volatility_regime: str           # HIGH/LOW/NORMAL
    breadth: float                   # % stocks above 50MA
    sector_rotation: str             # RISK_ON/RISK_OFF/NEUTRAL
    overall_score: float             # 0-100 market health


@dataclass
class StockOpportunity:
    """Complete opportunity assessment."""
    ticker: str
    name: str
    price: float
    
    # Flags
    flags: TechnicalFlags
    
    # Composite Ranking
    overall_rank: int                # 1 = best
    category: str                    # BREAKOUT_IMMINENT/ACCUMULATING/EARLY_STAGE/MOMENTUM
    
    # Trade Parameters
    entry_zone: Tuple[float, float]  # Entry price range
    stop_loss: float                 # Stop loss level
    target_1: float                  # First target
    target_2: float                  # Second target
    position_size_pct: float         # Suggested position size
    
    # Probability & Timing
    success_probability: float       # Estimated probability
    optimal_entry_window: str        # NOW/WAIT_PULLBACK/WAIT_BREAKOUT


# =============================================================================
# TECHNICAL ANALYSIS ENGINE
# =============================================================================

class TechnicalAnalyzer:
    """Comprehensive technical analysis for a single stock."""
    
    def __init__(self, df: pd.DataFrame, ticker: str, name: str):
        self.df = df.copy()
        self.ticker = ticker
        self.name = name
        self.close = df['close']
        self.high = df['high']
        self.low = df['low']
        self.volume = df['volume']
        
    # =========================================================================
    # CONSOLIDATION & BASE ANALYSIS
    # =========================================================================
    
    def calculate_consolidation(self, lookback: int = 20) -> Tuple[int, float, float]:
        """
        Identify consolidation patterns.
        Consolidation = price trading in a tight range.
        """
        recent = self.df.tail(lookback)
        
        # Calculate daily ranges
        high_low_range = (recent['high'] - recent['low']) / recent['close']
        avg_range = high_low_range.mean()
        
        # ATR for reference
        atr = self.calculate_atr(14)
        atr_pct = atr / self.close.iloc[-1]
        
        # Count days in consolidation (within 2 ATR range)
        price_range_high = recent['high'].max()
        price_range_low = recent['low'].min()
        total_range = (price_range_high - price_range_low) / price_range_low
        
        # Tightness score (lower = tighter)
        tightness = total_range / (atr_pct * lookback)
        
        # Days in consolidation (price within 5% range)
        consolidation_days = 0
        for i in range(len(self.df) - 1, max(0, len(self.df) - 60), -1):
            window = self.df.iloc[i-10:i+1] if i >= 10 else self.df.iloc[:i+1]
            if len(window) < 5:
                break
            w_range = (window['high'].max() - window['low'].min()) / window['close'].mean()
            if w_range < 0.08:  # 8% range
                consolidation_days += 1
            else:
                break
        
        # Breakout proximity (% to recent high)
        recent_high = self.df.tail(60)['high'].max()
        current = self.close.iloc[-1]
        breakout_proximity = (recent_high - current) / current * 100
        
        return consolidation_days, tightness, breakout_proximity
    
    def identify_base_pattern(self) -> Tuple[str, float, int]:
        """
        Identify base patterns (Cup & Handle, Flat Base, VCP, Flag).
        """
        lookback = min(120, len(self.df))
        recent = self.df.tail(lookback)
        
        if len(recent) < 30:
            return "NONE", 0, 0
        
        high_52w = recent['high'].max()
        low_52w = recent['low'].min()
        current = self.close.iloc[-1]
        
        # Base depth
        depth = (high_52w - low_52w) / high_52w * 100
        
        # Base length in weeks
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        try:
            high_pos = recent.index.get_loc(high_idx)
            low_pos = recent.index.get_loc(low_idx)
            base_days = abs(high_pos - low_pos)
            base_weeks = base_days // 5
        except:
            base_weeks = lookback // 5
        
        # Pattern identification
        pattern = "NONE"
        
        # VCP: Volatility contracts over time
        first_half = recent.iloc[:len(recent)//2]
        second_half = recent.iloc[len(recent)//2:]
        
        first_vol = (first_half['high'] - first_half['low']).mean()
        second_vol = (second_half['high'] - second_half['low']).mean()
        
        if second_vol < first_vol * 0.7:  # 30% contraction
            pattern = "VCP"
        
        # Cup pattern: U-shape
        elif low_pos > high_pos and current > low_52w * 1.15:
            # Check for handle
            last_10 = recent.tail(10)
            if (last_10['high'].max() - last_10['low'].min()) / last_10['close'].mean() < 0.05:
                pattern = "CUP_HANDLE"
            else:
                pattern = "CUP"
        
        # Flat base: Trading in tight range near highs
        elif depth < 15 and current > high_52w * 0.95:
            pattern = "FLAT_BASE"
        
        # Flag pattern: Tight consolidation after run-up
        elif depth < 20 and base_weeks <= 4:
            pattern = "FLAG"
        
        return pattern, depth, base_weeks
    
    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================
    
    def calculate_atr(self, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = self.high
        low = self.low
        close = self.close
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def calculate_vcp_score(self) -> float:
        """
        Calculate Volatility Contraction Pattern score.
        Higher score = better VCP setup.
        """
        # Look for 3 contractions
        contractions = []
        
        for i in range(3):
            start = -60 + i * 15
            end = -45 + i * 15 if i < 2 else None
            
            if end:
                window = self.df.iloc[start:end]
            else:
                window = self.df.iloc[start:]
            
            if len(window) > 5:
                volatility = (window['high'] - window['low']).mean() / window['close'].mean()
                contractions.append(volatility)
        
        if len(contractions) < 2:
            return 0
        
        # Check for contraction pattern
        contracting = all(contractions[i] > contractions[i+1] for i in range(len(contractions)-1))
        
        if contracting:
            contraction_rate = (contractions[0] - contractions[-1]) / contractions[0]
            return min(100, contraction_rate * 150)
        
        return 0
    
    def check_bb_squeeze(self) -> Tuple[bool, int]:
        """
        Check for Bollinger Band squeeze (low volatility before expansion).
        """
        close = self.close
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_width = (2 * std_20) / sma_20
        
        # Keltner Channel for comparison
        atr = self.calculate_atr(20)
        kc_width = (2 * 1.5 * atr) / sma_20.iloc[-1]
        
        # Squeeze = BB inside KC
        current_bb_width = bb_width.iloc[-1]
        squeeze = current_bb_width < kc_width
        
        # Count squeeze days
        squeeze_days = 0
        for i in range(len(bb_width) - 1, max(0, len(bb_width) - 30), -1):
            if bb_width.iloc[i] < kc_width:
                squeeze_days += 1
            else:
                break
        
        return squeeze, squeeze_days
    
    # =========================================================================
    # MOMENTUM & RELATIVE STRENGTH
    # =========================================================================
    
    def calculate_momentum(self) -> Tuple[float, float, float]:
        """Calculate momentum over multiple timeframes."""
        mom_20d = self.close.pct_change(20).iloc[-1] * 100 if len(self.close) > 20 else 0
        mom_60d = self.close.pct_change(60).iloc[-1] * 100 if len(self.close) > 60 else 0
        
        # Momentum acceleration
        mom_10d = self.close.pct_change(10).iloc[-1] if len(self.close) > 10 else 0
        mom_10d_prev = self.close.pct_change(10).iloc[-11] if len(self.close) > 21 else 0
        accel = (mom_10d - mom_10d_prev) * 100
        
        return mom_20d, mom_60d, accel
    
    # =========================================================================
    # VOLUME ANALYSIS
    # =========================================================================
    
    def calculate_accumulation(self) -> Tuple[float, str, str]:
        """
        Calculate accumulation/distribution.
        Based on price location in range and volume.
        """
        # Money Flow Multiplier
        mfm = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 0.001)
        mfv = mfm * self.volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        
        # Recent trend
        ad_20d = ad_line.tail(20)
        ad_slope = (ad_20d.iloc[-1] - ad_20d.iloc[0]) / len(ad_20d)
        
        # Normalize to score
        vol_avg = self.volume.tail(60).mean()
        score = ad_slope / vol_avg * 1000
        score = max(-100, min(100, score))
        
        # Determine trend
        if score > 20:
            trend = "ACCUMULATING"
        elif score < -20:
            trend = "DISTRIBUTING"
        else:
            trend = "NEUTRAL"
        
        # Check for divergence
        price_trend = self.close.tail(20).iloc[-1] - self.close.tail(20).iloc[0]
        ad_trend = ad_20d.iloc[-1] - ad_20d.iloc[0]
        
        if price_trend < 0 and ad_trend > 0:
            divergence = "BULLISH_DIV"
        elif price_trend > 0 and ad_trend < 0:
            divergence = "BEARISH_DIV"
        else:
            divergence = "NONE"
        
        return score, trend, divergence
    
    def check_volume_dry_up(self) -> bool:
        """Check if volume is drying up (bullish during base building)."""
        vol_10d = self.volume.tail(10).mean()
        vol_50d = self.volume.tail(50).mean()
        
        return vol_10d < vol_50d * 0.7
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    
    def analyze_trend(self) -> Tuple[bool, bool, bool, str, float]:
        """Analyze trend using moving averages."""
        close = self.close
        
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma_50
        
        current = close.iloc[-1]
        
        above_20 = current > ma_20
        above_50 = current > ma_50
        above_200 = current > ma_200
        
        # MA alignment
        if ma_20 > ma_50 > ma_200:
            alignment = "BULLISH"
        elif ma_20 < ma_50 < ma_200:
            alignment = "BEARISH"
        else:
            alignment = "MIXED"
        
        # Trend strength (ADX)
        adx = self.calculate_adx()
        
        return above_20, above_50, above_200, alignment, adx
    
    def calculate_adx(self, period: int = 14) -> float:
        """Calculate ADX for trend strength."""
        try:
            high = self.high
            low = self.low
            close = self.close
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return adx.iloc[-1]
        except:
            return 25
    
    def determine_stage(self) -> str:
        """
        Determine Weinstein Stage:
        Stage 1: Basing (accumulation)
        Stage 2: Advancing (markup)
        Stage 3: Topping (distribution)
        Stage 4: Declining (markdown)
        """
        close = self.close
        ma_30 = close.rolling(30).mean()
        ma_30_slope = (ma_30.iloc[-1] - ma_30.iloc[-20]) / ma_30.iloc[-20] * 100
        
        current = close.iloc[-1]
        above_ma = current > ma_30.iloc[-1]
        
        # Volume analysis
        vol_trend = self.volume.tail(20).mean() / self.volume.tail(60).mean()
        
        if not above_ma and abs(ma_30_slope) < 2:
            return "STAGE_1"  # Basing
        elif above_ma and ma_30_slope > 2:
            return "STAGE_2"  # Advancing - BUY
        elif above_ma and ma_30_slope < 2 and vol_trend > 1.2:
            return "STAGE_3"  # Topping - SELL
        elif not above_ma and ma_30_slope < -2:
            return "STAGE_4"  # Declining - AVOID
        else:
            return "TRANSITION"
    
    # =========================================================================
    # COMPOSITE ANALYSIS
    # =========================================================================
    
    def analyze(self) -> TechnicalFlags:
        """Run complete technical analysis and generate flags."""
        
        # Consolidation
        consol_days, tightness, breakout_prox = self.calculate_consolidation()
        
        # Base pattern
        pattern, depth, base_weeks = self.identify_base_pattern()
        
        # Volatility
        vcp_score = self.calculate_vcp_score()
        squeeze, squeeze_days = self.check_bb_squeeze()
        atr = self.calculate_atr(14)
        atr_20 = self.calculate_atr(20)
        atr_contraction = atr / atr_20 if atr_20 > 0 else 1
        
        # Momentum
        mom_20d, mom_60d, mom_accel = self.calculate_momentum()
        
        # Volume
        accum_score, vol_trend, obv_div = self.calculate_accumulation()
        vol_dry = self.check_volume_dry_up()
        
        # Trend
        above_20, above_50, above_200, ma_align, adx = self.analyze_trend()
        
        # Stage
        stage = self.determine_stage()
        
        # Handle forming check
        handle = pattern == "CUP_HANDLE"
        
        # === COMPOSITE SCORES ===
        
        # Setup Quality (how good is the setup)
        setup_score = 0
        if pattern in ["VCP", "CUP_HANDLE", "FLAT_BASE"]:
            setup_score += 30
        if squeeze and squeeze_days >= 5:
            setup_score += 20
        if vcp_score > 50:
            setup_score += 15
        if vol_trend == "ACCUMULATING":
            setup_score += 15
        if stage == "STAGE_2":
            setup_score += 10
        if ma_align == "BULLISH":
            setup_score += 10
        
        # Timing Score (how good is timing now)
        timing_score = 0
        if breakout_prox < 3:  # Within 3% of breakout
            timing_score += 30
        if squeeze:
            timing_score += 20
        if vol_dry:
            timing_score += 15
        if obv_div == "BULLISH_DIV":
            timing_score += 20
        if mom_accel > 0:
            timing_score += 15
        
        # Risk/Reward estimate
        if depth > 0:
            risk_reward = min(5, 15 / max(5, depth))  # Shallower base = better R/R
        else:
            risk_reward = 2.0
        
        return TechnicalFlags(
            ticker=self.ticker,
            name=self.name,
            consolidation_days=consol_days,
            consolidation_tightness=round(tightness, 2),
            breakout_proximity=round(breakout_prox, 2),
            stage=stage,
            vcp_score=round(vcp_score, 1),
            bb_squeeze=squeeze,
            bb_squeeze_days=squeeze_days,
            atr_contraction=round(atr_contraction, 2),
            rs_rank=0,  # Will be set by scanner
            rs_new_high=False,
            momentum_20d=round(mom_20d, 2),
            momentum_60d=round(mom_60d, 2),
            momentum_accel=round(mom_accel, 2),
            accumulation_score=round(accum_score, 1),
            volume_trend=vol_trend,
            obv_divergence=obv_div,
            volume_dry_up=vol_dry,
            above_20ma=above_20,
            above_50ma=above_50,
            above_200ma=above_200,
            ma_alignment=ma_align,
            trend_strength=round(adx, 1),
            base_pattern=pattern,
            base_depth=round(depth, 1),
            base_length_weeks=base_weeks,
            handle_forming=handle,
            setup_quality=round(setup_score, 1),
            timing_score=round(timing_score, 1),
            risk_reward=round(risk_reward, 2)
        )


# =============================================================================
# MARKET REGIME ANALYZER
# =============================================================================

class MarketRegimeAnalyzer:
    """Analyze overall market conditions."""
    
    def __init__(self, all_data: Dict[str, pd.DataFrame]):
        self.all_data = all_data
    
    def analyze(self) -> MarketRegime:
        """Determine current market regime."""
        
        # Calculate breadth
        above_50ma = 0
        total = 0
        
        for ticker, df in self.all_data.items():
            if len(df) >= 50:
                close = df['close']
                ma_50 = close.rolling(50).mean().iloc[-1]
                if close.iloc[-1] > ma_50:
                    above_50ma += 1
                total += 1
        
        breadth = above_50ma / max(1, total) * 100
        
        # Market trend (use largest stocks as proxy)
        major_tickers = ['2222.SR', '1180.SR', '7010.SR', '2010.SR', '1010.SR']
        trend_scores = []
        
        for ticker in major_tickers:
            if ticker in self.all_data:
                df = self.all_data[ticker]
                close = df['close']
                ma_50 = close.rolling(50).mean().iloc[-1]
                ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma_50
                
                if close.iloc[-1] > ma_50 > ma_200:
                    trend_scores.append(100)
                elif close.iloc[-1] > ma_50:
                    trend_scores.append(70)
                elif close.iloc[-1] > ma_200:
                    trend_scores.append(40)
                else:
                    trend_scores.append(20)
        
        avg_trend = np.mean(trend_scores) if trend_scores else 50
        
        if avg_trend > 70:
            trend = "BULL"
        elif avg_trend < 40:
            trend = "BEAR"
        else:
            trend = "NEUTRAL"
        
        # Volatility regime
        volatilities = []
        for ticker, df in list(self.all_data.items())[:20]:
            if len(df) >= 20:
                ret = df['close'].pct_change().tail(20).std() * np.sqrt(252)
                volatilities.append(ret)
        
        avg_vol = np.mean(volatilities) if volatilities else 0.3
        
        if avg_vol > 0.4:
            vol_regime = "HIGH"
        elif avg_vol < 0.2:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"
        
        # Sector rotation (simplified)
        if breadth > 60:
            rotation = "RISK_ON"
        elif breadth < 40:
            rotation = "RISK_OFF"
        else:
            rotation = "NEUTRAL"
        
        # Overall market health score
        health = (breadth * 0.4 + avg_trend * 0.4 + (50 if vol_regime == "NORMAL" else 30) * 0.2)
        
        return MarketRegime(
            trend=trend,
            trend_strength=round(avg_trend, 1),
            volatility_regime=vol_regime,
            breadth=round(breadth, 1),
            sector_rotation=rotation,
            overall_score=round(health, 1)
        )


# =============================================================================
# CHAMPIONSHIP SCANNER
# =============================================================================

class ChampionshipScanner:
    """
    The main scanner combining all analysis.
    Identifies the best opportunities based on championship-level criteria.
    """
    
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.stock_names: Dict[str, str] = {}
        self.all_flags: Dict[str, TechnicalFlags] = {}
        self.market_regime: Optional[MarketRegime] = None
        self.opportunities: List[StockOpportunity] = []
        self.scan_time = None
        self.history_file = "output/production/flag_history.json"
        self.flag_history: Dict[str, List[dict]] = {}
    
    def fetch_data(self) -> None:
        """Fetch all stock data."""
        total = len(TASI_TICKERS)
        print(f"Fetching data for {total} TASI tickers...")
        
        for i, ticker in enumerate(TASI_TICKERS):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")
                
                if len(df) >= 60:
                    df.columns = [c.lower() for c in df.columns]
                    self.stock_data[ticker] = df
                    
                    try:
                        info = stock.info
                        name = info.get('longName') or info.get('shortName') or ticker
                        self.stock_names[ticker] = name[:40]
                    except:
                        self.stock_names[ticker] = ticker
            except:
                pass
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{total}")
        
        print(f"Loaded {len(self.stock_data)} stocks")
    
    def analyze_all_stocks(self) -> None:
        """Run technical analysis on all stocks."""
        print("\nAnalyzing all stocks...")
        
        for ticker, df in self.stock_data.items():
            name = self.stock_names.get(ticker, ticker)
            analyzer = TechnicalAnalyzer(df, ticker, name)
            self.all_flags[ticker] = analyzer.analyze()
        
        # Calculate RS rankings
        momentums = [(t, f.momentum_60d) for t, f in self.all_flags.items()]
        momentums.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (ticker, _) in enumerate(momentums):
            percentile = int((1 - rank / len(momentums)) * 100)
            self.all_flags[ticker].rs_rank = percentile
            
            # Check if RS at new high
            if percentile >= 90:
                self.all_flags[ticker].rs_new_high = True
    
    def analyze_market(self) -> None:
        """Analyze market regime."""
        analyzer = MarketRegimeAnalyzer(self.stock_data)
        self.market_regime = analyzer.analyze()
    
    def load_history(self) -> None:
        """Load flag history for time-series analysis."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.flag_history = json.load(f)
            except:
                pass
    
    def save_history(self) -> None:
        """Save current flags to history."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        for ticker, flags in self.all_flags.items():
            if ticker not in self.flag_history:
                self.flag_history[ticker] = []
            
            # Keep last 30 days
            self.flag_history[ticker] = self.flag_history[ticker][-29:]
            
            self.flag_history[ticker].append({
                'date': today,
                'consolidation_days': int(flags.consolidation_days),
                'vcp_score': float(flags.vcp_score),
                'bb_squeeze': bool(flags.bb_squeeze),
                'rs_rank': int(flags.rs_rank),
                'accumulation_score': float(flags.accumulation_score),
                'stage': str(flags.stage),
                'setup_quality': float(flags.setup_quality)
            })
        
        with open(self.history_file, 'w') as f:
            json.dump(self.flag_history, f)
    
    def analyze_history_patterns(self, ticker: str) -> dict:
        """Analyze historical flag patterns for a stock."""
        if ticker not in self.flag_history or len(self.flag_history[ticker]) < 5:
            return {}
        
        history = self.flag_history[ticker]
        
        # Check for improving patterns
        recent = history[-5:]
        
        # Consolidation trend
        consol_trend = recent[-1].get('consolidation_days', 0) - recent[0].get('consolidation_days', 0)
        
        # VCP building
        vcp_improving = all(
            recent[i].get('vcp_score', 0) <= recent[i+1].get('vcp_score', 0) 
            for i in range(len(recent)-1)
        )
        
        # Squeeze persistence
        squeeze_days = sum(1 for r in recent if r.get('bb_squeeze', False))
        
        # RS rank improvement
        rs_improving = recent[-1].get('rs_rank', 0) > recent[0].get('rs_rank', 0)
        
        return {
            'consolidation_building': consol_trend > 3,
            'vcp_improving': vcp_improving,
            'squeeze_persistent': squeeze_days >= 3,
            'rs_improving': rs_improving,
            'setup_building': recent[-1].get('setup_quality', 0) > recent[0].get('setup_quality', 0)
        }
    
    def rank_opportunities(self) -> None:
        """Rank all stocks and identify best opportunities."""
        
        candidates = []
        
        for ticker, flags in self.all_flags.items():
            df = self.stock_data[ticker]
            price = df['close'].iloc[-1]
            
            # Calculate opportunity score
            score = 0
            category = "WATCHING"
            
            # === BREAKOUT IMMINENT ===
            if (flags.bb_squeeze and flags.bb_squeeze_days >= 5 and 
                flags.breakout_proximity < 5 and flags.volume_trend in ["ACCUMULATING", "NEUTRAL"]):
                score += 40
                category = "BREAKOUT_IMMINENT"
            
            # === STRONG ACCUMULATION ===
            if (flags.accumulation_score > 30 and flags.stage in ["STAGE_1", "STAGE_2"] and
                flags.obv_divergence == "BULLISH_DIV"):
                score += 35
                if category == "WATCHING":
                    category = "ACCUMULATING"
            
            # === VCP SETUP ===
            if flags.vcp_score > 60 and flags.base_pattern == "VCP":
                score += 30
                if category == "WATCHING":
                    category = "VCP_SETUP"
            
            # === EARLY STAGE 2 ===
            if (flags.stage == "STAGE_2" and flags.ma_alignment == "BULLISH" and
                flags.momentum_20d > 0 and flags.rs_rank > 70):
                score += 25
                if category == "WATCHING":
                    category = "EARLY_STAGE"
            
            # === MOMENTUM LEADER ===
            if flags.rs_rank >= 90 and flags.momentum_accel > 0:
                score += 20
                if category == "WATCHING":
                    category = "MOMENTUM"
            
            # Add component scores
            score += flags.setup_quality * 0.3
            score += flags.timing_score * 0.3
            score += flags.rs_rank * 0.1
            
            # Market regime filter
            if self.market_regime.trend == "BEAR" and category not in ["ACCUMULATING"]:
                score *= 0.5
            
            # Historical pattern bonus
            hist_patterns = self.analyze_history_patterns(ticker)
            if hist_patterns.get('vcp_improving'):
                score += 10
            if hist_patterns.get('squeeze_persistent'):
                score += 8
            if hist_patterns.get('setup_building'):
                score += 5
            
            # Calculate trade parameters
            atr = df['high'].tail(14).values - df['low'].tail(14).values
            atr = np.mean(atr)
            
            recent_high = df['high'].tail(60).max()
            recent_low = df['low'].tail(20).min()
            
            stop_loss = price - (2 * atr)
            target_1 = price + (3 * atr)
            target_2 = price + (5 * atr)
            
            # Entry window
            if flags.breakout_proximity < 2:
                entry_window = "NOW"
            elif flags.bb_squeeze:
                entry_window = "WAIT_BREAKOUT"
            else:
                entry_window = "WAIT_PULLBACK"
            
            # Success probability estimate
            prob = 0.5
            if flags.stage == "STAGE_2":
                prob += 0.1
            if flags.ma_alignment == "BULLISH":
                prob += 0.1
            if flags.rs_rank > 80:
                prob += 0.1
            if flags.accumulation_score > 20:
                prob += 0.05
            if self.market_regime.trend == "BULL":
                prob += 0.1
            
            prob = min(0.85, prob)
            
            candidates.append(StockOpportunity(
                ticker=ticker,
                name=self.stock_names.get(ticker, ticker),
                price=round(price, 2),
                flags=flags,
                overall_rank=0,
                category=category,
                entry_zone=(round(price * 0.98, 2), round(price * 1.02, 2)),
                stop_loss=round(stop_loss, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                position_size_pct=round(min(10, 100 / max(1, flags.base_depth)) * 0.5, 1),
                success_probability=round(prob, 2),
                optimal_entry_window=entry_window
            ))
            
            candidates[-1]._score = score
        
        # Sort and rank
        candidates.sort(key=lambda x: x._score, reverse=True)
        for i, c in enumerate(candidates):
            c.overall_rank = i + 1
        
        self.opportunities = candidates
    
    def run(self) -> str:
        """Run complete analysis."""
        self.scan_time = datetime.now()
        
        self.load_history()
        self.fetch_data()
        self.analyze_all_stocks()
        self.analyze_market()
        self.rank_opportunities()
        self.save_history()
        
        return self.generate_report()
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        lines = []
        
        lines.append("=" * 120)
        lines.append("🏆 CHAMPIONSHIP TRADING SCANNER - TASI MARKET")
        lines.append(f"   Generated: {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("   Strategy: Minervini VCP + O'Neil RS + Weinstein Stages")
        lines.append("=" * 120)
        
        # Market Regime
        mr = self.market_regime
        regime_emoji = "🟢" if mr.trend == "BULL" else "🔴" if mr.trend == "BEAR" else "🟡"
        
        lines.append(f"""
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MARKET REGIME                                                                                                         │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  {regime_emoji} Trend: {mr.trend:<10} Strength: {mr.trend_strength:>5.1f}  │  Volatility: {mr.volatility_regime:<8}  │  Breadth: {mr.breadth:>5.1f}%  │  {mr.sector_rotation:<10}  │
│  Overall Market Health: {mr.overall_score:.1f}/100                                                                                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
""")
        
        # Top Opportunities by Category
        categories = ["BREAKOUT_IMMINENT", "VCP_SETUP", "ACCUMULATING", "EARLY_STAGE", "MOMENTUM"]
        
        for cat in categories:
            cat_opps = [o for o in self.opportunities if o.category == cat][:5]
            
            if not cat_opps:
                continue
            
            cat_emoji = {
                "BREAKOUT_IMMINENT": "🚀",
                "VCP_SETUP": "📐",
                "ACCUMULATING": "💰",
                "EARLY_STAGE": "🌱",
                "MOMENTUM": "⚡"
            }.get(cat, "📊")
            
            lines.append(f"\n{'='*120}")
            lines.append(f"{cat_emoji} {cat.replace('_', ' ')} - Top {len(cat_opps)} Setups")
            lines.append("=" * 120)
            
            lines.append(f"\n  {'Rank':<5} {'Ticker':<10} {'Company':<30} {'Price':>8} {'Setup':>6} {'Timing':>6} {'RS':>4} {'Entry Window':<15}")
            lines.append("  " + "-" * 115)
            
            for o in cat_opps:
                f = o.flags
                lines.append(f"  {o.overall_rank:<5} {o.ticker:<10} {o.name[:30]:<30} {o.price:>8.2f} "
                           f"{f.setup_quality:>5.0f} {f.timing_score:>6.0f} {f.rs_rank:>4} {o.optimal_entry_window:<15}")
        
        # Detailed Top 10 Opportunities
        lines.append(f"""

{'='*120}
📋 DETAILED TOP 10 OPPORTUNITIES
{'='*120}
""")
        
        for o in self.opportunities[:10]:
            f = o.flags
            
            # Build flag string
            flags_str = []
            if f.bb_squeeze:
                flags_str.append(f"🔥 BB Squeeze ({f.bb_squeeze_days}d)")
            if f.vcp_score > 50:
                flags_str.append(f"📐 VCP ({f.vcp_score:.0f})")
            if f.volume_trend == "ACCUMULATING":
                flags_str.append("💰 Accumulating")
            if f.obv_divergence == "BULLISH_DIV":
                flags_str.append("📈 Bullish Divergence")
            if f.rs_rank >= 90:
                flags_str.append(f"⚡ RS Leader ({f.rs_rank})")
            if f.base_pattern != "NONE":
                flags_str.append(f"📊 {f.base_pattern}")
            
            lines.append(f"""
  #{o.overall_rank} {o.ticker} - {o.name}
  {'─'*110}
  Category: {o.category} | Stage: {f.stage} | MA Alignment: {f.ma_alignment}
  
  CURRENT STATUS:
    Price:              {o.price:.2f} SAR
    RS Rank:            {f.rs_rank}/100 {'🔥 TOP DECILE' if f.rs_rank >= 90 else ''}
    Setup Quality:      {f.setup_quality:.0f}/100
    Timing Score:       {f.timing_score:.0f}/100
  
  ACTIVE FLAGS:
    {' | '.join(flags_str) if flags_str else 'None'}
  
  CONSOLIDATION:
    Days in Range:      {f.consolidation_days}
    Base Pattern:       {f.base_pattern}
    Base Depth:         {f.base_depth:.1f}%
    Breakout Distance:  {f.breakout_proximity:.1f}%
  
  VOLUME ANALYSIS:
    Volume Trend:       {f.volume_trend}
    Accum Score:        {f.accumulation_score:.1f}
    Volume Dry-Up:      {'Yes ✓' if f.volume_dry_up else 'No'}
  
  TRADE PARAMETERS:
    Entry Zone:         {o.entry_zone[0]:.2f} - {o.entry_zone[1]:.2f} SAR
    Stop Loss:          {o.stop_loss:.2f} SAR ({(o.stop_loss/o.price-1)*100:.1f}%)
    Target 1:           {o.target_1:.2f} SAR ({(o.target_1/o.price-1)*100:.1f}%)
    Target 2:           {o.target_2:.2f} SAR ({(o.target_2/o.price-1)*100:.1f}%)
    Position Size:      {o.position_size_pct:.1f}% of portfolio
    Success Prob:       {o.success_probability:.0%}
    Entry Window:       {o.optimal_entry_window}
""")
        
        # Summary Stats
        breakout_count = len([o for o in self.opportunities if o.category == "BREAKOUT_IMMINENT"])
        vcp_count = len([o for o in self.opportunities if o.category == "VCP_SETUP"])
        accum_count = len([o for o in self.opportunities if o.category == "ACCUMULATING"])
        
        lines.append(f"""
{'='*120}
📊 SCAN SUMMARY
{'='*120}

  Stocks Scanned:           {len(self.stock_data)}
  Breakout Imminent:        {breakout_count}
  VCP Setups:               {vcp_count}
  Accumulating:             {accum_count}
  
  Market Regime:            {mr.trend} ({mr.trend_strength:.0f}/100)
  Recommended Exposure:     {'FULL' if mr.trend == 'BULL' else 'REDUCED' if mr.trend == 'BEAR' else 'MODERATE'}
  
{'='*120}
END OF REPORT
{'='*120}
""")
        
        return "\n".join(lines)
    
    def save_report(self, output_dir: str = "output/production") -> str:
        """Save report to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.scan_time.strftime('%Y%m%d_%H%M%S')
        filepath = f"{output_dir}/championship_scan_{timestamp}.txt"
        
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        
        with open(f"{output_dir}/latest_championship_scan.txt", 'w') as f:
            f.write(report)
        
        return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🏆 CHAMPIONSHIP TRADING SCANNER")
    print("   Combining Minervini VCP + O'Neil RS + Weinstein Stages")
    print("=" * 70 + "\n")
    
    scanner = ChampionshipScanner(capital=1_000_000)
    report = scanner.run()
    print(report)
    filepath = scanner.save_report()
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
