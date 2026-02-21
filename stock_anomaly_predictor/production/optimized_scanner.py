#!/usr/bin/env python3
"""
OPTIMIZED BLENDED STRATEGY SCANNER
==================================
Based on 2019-2026 backtest results across 103 TASI stocks.

WINNING STRATEGIES (to keep):
1. Cup & Handle (PF=1.80, Win=54.2%) - BEST
2. Volume Dry-Up Breakout (PF=1.76, Win=54.8%)
3. Accumulation Breakout (PF=1.74, Win=54.0%)
4. Flat Base Breakout (PF=1.65, Win=53.1%)
5. BB Squeeze Breakout (PF=1.47, Win=50.0%)
6. RS Momentum (PF=1.37, Win=48.8%)

DROPPED STRATEGIES (don't use):
- VCP Pattern (no signals generated)
- MA Alignment (no signals generated)
- Stage 2 Entry (weak PF=1.12)

BLENDED SCORING:
Each stock gets a composite score based on which strategies are active.
Higher score = more strategies confirming = higher probability trade.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


# =============================================================================
# STRATEGY WEIGHTS (From Backtest Results)
# =============================================================================

STRATEGY_WEIGHTS = {
    'cup_handle': 1.80,           # PF=1.80, Win=54.2%
    'volume_dryup': 1.76,         # PF=1.76, Win=54.8%
    'accumulation': 1.74,         # PF=1.74, Win=54.0%
    'flat_base': 1.65,            # PF=1.65, Win=53.1%
    'bb_squeeze': 1.47,           # PF=1.47, Win=50.0%
    'rs_momentum': 1.37,          # PF=1.37, Win=48.8%
}

TOTAL_WEIGHT = sum(STRATEGY_WEIGHTS.values())


# =============================================================================
# TASI TICKERS
# =============================================================================

TASI_TICKERS = [
    '1180.SR', '1010.SR', '1150.SR', '1140.SR', '1060.SR', '1050.SR',
    '1020.SR', '1030.SR', '1080.SR', '2222.SR', '4030.SR', '2381.SR',
    '2380.SR', '2010.SR', '2290.SR', '2250.SR', '2210.SR', '2001.SR',
    '2060.SR', '2310.SR', '2350.SR', '2330.SR', '2170.SR', '2020.SR',
    '2190.SR', '1211.SR', '1320.SR', '1321.SR', '1302.SR', '1304.SR',
    '2200.SR', '2220.SR', '2240.SR', '2320.SR', '2370.SR', '3010.SR',
    '3020.SR', '3030.SR', '3040.SR', '3050.SR', '3060.SR', '3080.SR',
    '3090.SR', '5110.SR', '2082.SR', '2083.SR', '4190.SR', '4003.SR',
    '4240.SR', '4001.SR', '4002.SR', '4004.SR', '4007.SR', '4009.SR',
    '4020.SR', '4031.SR', '4050.SR', '4080.SR', '4110.SR', '4140.SR',
    '4200.SR', '4220.SR', '4250.SR', '4270.SR', '4280.SR', '4290.SR',
    '2280.SR', '2050.SR', '6002.SR', '6001.SR', '6010.SR', '4071.SR',
    '7010.SR', '7020.SR', '7030.SR', '4300.SR', '4310.SR', '4320.SR',
    '4330.SR', '4331.SR', '4332.SR', '4333.SR', '1120.SR', '8010.SR',
    '8012.SR', '8020.SR', '8030.SR', '8040.SR', '8050.SR', '8060.SR',
    '8100.SR', '8120.SR', '8150.SR', '8160.SR', '8180.SR', '8200.SR',
    '8210.SR', '8230.SR', '8240.SR', '8250.SR', '8300.SR', '8310.SR',
]


# =============================================================================
# SIGNAL DATA
# =============================================================================

@dataclass
class BlendedSignal:
    ticker: str
    name: str
    price: float
    
    # Individual Strategy Signals (True/False)
    cup_handle: bool
    volume_dryup: bool
    accumulation: bool
    flat_base: bool
    bb_squeeze: bool
    rs_momentum: bool
    
    # Composite
    active_strategies: List[str]
    blended_score: float
    signal_strength: str  # STRONG/MODERATE/WEAK
    
    # Trade Parameters
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    position_size_pct: float
    
    # Probability
    expected_win_rate: float


# =============================================================================
# STRATEGY DETECTORS
# =============================================================================

class StrategyDetector:
    """Detect signals for each proven strategy."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['close']
        self.high = df['high']
        self.low = df['low']
        self.volume = df['volume']
    
    def detect_cup_handle(self) -> bool:
        """
        Cup & Handle Pattern - BEST STRATEGY (PF=1.80)
        U-shaped base, handle consolidation, then breakout.
        """
        if len(self.df) < 60:
            return False
        
        # 60-day lookback for cup
        rolling_high = self.high.tail(60).max()
        rolling_low = self.close.tail(60).min()
        
        # Cup depth 15-35%
        cup_depth = (rolling_high - rolling_low) / rolling_high
        if not (0.15 < cup_depth < 0.35):
            return False
        
        # Current price near cup high (within 10%)
        current = self.close.iloc[-1]
        if current < rolling_high * 0.90:
            return False
        
        # Handle: small recent consolidation
        recent_range = (self.high.tail(10).max() - self.low.tail(10).min()) / current
        if recent_range > 0.08:
            return False
        
        # Near breakout (within 5%)
        if current < rolling_high * 0.95:
            return False
        
        return True
    
    def detect_volume_dryup(self) -> bool:
        """
        Volume Dry-Up Breakout - 2nd BEST (PF=1.76)
        Volume contracts during consolidation, ready to expand.
        """
        if len(self.df) < 50:
            return False
        
        # Volume dry-up
        vol_ma = self.volume.rolling(50).mean()
        vol_ratio = self.volume.tail(10).mean() / vol_ma.iloc[-1]
        
        if vol_ratio > 0.7:  # Volume should be below 70% of average
            return False
        
        # Price near highs
        recent_high = self.high.tail(40).max()
        current = self.close.iloc[-1]
        
        if current < recent_high * 0.95:
            return False
        
        return True
    
    def detect_accumulation(self) -> bool:
        """
        Accumulation Breakout - 3rd BEST (PF=1.74)
        A/D line rising while price consolidates.
        """
        if len(self.df) < 30:
            return False
        
        # Accumulation/Distribution
        mfm = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 0.001)
        mfv = mfm * self.volume
        ad_line = mfv.cumsum()
        
        # A/D slope positive
        ad_slope = ad_line.iloc[-1] - ad_line.iloc[-20]
        if ad_slope <= 0:
            return False
        
        # Price consolidating (tight range)
        price_range = (self.high.tail(20).max() - self.low.tail(20).min()) / self.close.iloc[-1]
        if price_range > 0.12:
            return False
        
        # Near recent high
        recent_high = self.high.tail(30).max()
        if self.close.iloc[-1] < recent_high * 0.95:
            return False
        
        return True
    
    def detect_flat_base(self) -> bool:
        """
        Flat Base Breakout - (PF=1.65)
        Tight consolidation near highs.
        """
        if len(self.df) < 30:
            return False
        
        # Base range
        base_high = self.high.tail(30).max()
        base_low = self.low.tail(30).min()
        base_range = (base_high - base_low) / base_low
        
        # Flat base < 15%
        if base_range > 0.15:
            return False
        
        # Near top of base
        current = self.close.iloc[-1]
        if current < base_high * 0.97:
            return False
        
        return True
    
    def detect_bb_squeeze(self) -> bool:
        """
        BB Squeeze Breakout - (PF=1.47)
        Bollinger Bands inside Keltner Channel.
        """
        if len(self.df) < 20:
            return False
        
        # Bollinger Band width
        sma = self.close.rolling(20).mean()
        std = self.close.rolling(20).std()
        bb_width = (2 * std) / sma
        
        # Keltner width
        tr = pd.concat([
            self.high - self.low,
            abs(self.high - self.close.shift(1)),
            abs(self.low - self.close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        kc_width = (2 * 1.5 * atr) / sma
        
        # Squeeze: BB inside KC for 5+ days
        squeeze = bb_width < kc_width
        squeeze_duration = squeeze.tail(10).sum()
        
        if squeeze_duration < 5:
            return False
        
        # Currently in squeeze or just released
        current_squeeze = squeeze.iloc[-1]
        
        # Price above upper BB (breakout)
        upper_bb = sma + (2 * std)
        breakout = self.close.iloc[-1] > upper_bb.iloc[-1]
        
        return current_squeeze or breakout
    
    def detect_rs_momentum(self) -> bool:
        """
        RS Momentum - (PF=1.37)
        Strong 60-day momentum.
        """
        if len(self.df) < 60:
            return False
        
        # 60-day momentum > 15%
        momentum = self.close.pct_change(60).iloc[-1]
        
        if momentum < 0.15:
            return False
        
        # Above 50 MA
        ma_50 = self.close.rolling(50).mean().iloc[-1]
        if self.close.iloc[-1] < ma_50:
            return False
        
        return True
    
    def detect_all(self) -> Dict[str, bool]:
        """Detect all strategies."""
        return {
            'cup_handle': self.detect_cup_handle(),
            'volume_dryup': self.detect_volume_dryup(),
            'accumulation': self.detect_accumulation(),
            'flat_base': self.detect_flat_base(),
            'bb_squeeze': self.detect_bb_squeeze(),
            'rs_momentum': self.detect_rs_momentum(),
        }


# =============================================================================
# OPTIMIZED SCANNER
# =============================================================================

class OptimizedScanner:
    """Scan for blended strategy signals."""
    
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.stock_names: Dict[str, str] = {}
        self.signals: List[BlendedSignal] = []
        self.scan_time = None
    
    def fetch_data(self) -> None:
        """Fetch stock data."""
        print(f"Fetching data for {len(TASI_TICKERS)} stocks...")
        
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
                print(f"  Progress: {i+1}/{len(TASI_TICKERS)}")
        
        print(f"Loaded {len(self.stock_data)} stocks")
    
    def calculate_trade_params(self, df: pd.DataFrame) -> Tuple[float, float, float, float]:
        """Calculate trade parameters."""
        close = df['close'].iloc[-1]
        atr = (df['high'] - df['low']).tail(14).mean()
        
        stop_loss = close - (2 * atr)
        target_1 = close * 1.15  # 15% target (from backtest)
        target_2 = close * 1.25  # 25% stretch target
        
        risk = close - stop_loss
        reward = target_1 - close
        risk_reward = reward / risk if risk > 0 else 0
        
        return stop_loss, target_1, target_2, risk_reward
    
    def scan_all(self) -> None:
        """Scan all stocks for signals."""
        print("\nScanning for blended signals...")
        
        for ticker, df in self.stock_data.items():
            detector = StrategyDetector(df)
            signals = detector.detect_all()
            
            # Count active strategies
            active = [name for name, is_active in signals.items() if is_active]
            
            if not active:
                continue
            
            # Calculate blended score
            blended_score = sum(STRATEGY_WEIGHTS[s] for s in active)
            normalized_score = (blended_score / TOTAL_WEIGHT) * 100
            
            # Signal strength
            if len(active) >= 4:
                strength = "STRONG"
            elif len(active) >= 2:
                strength = "MODERATE"
            else:
                strength = "WEAK"
            
            # Trade parameters
            stop_loss, target_1, target_2, rr = self.calculate_trade_params(df)
            
            # Expected win rate (weighted average of active strategies)
            win_rates = {
                'cup_handle': 54.2,
                'volume_dryup': 54.8,
                'accumulation': 54.0,
                'flat_base': 53.1,
                'bb_squeeze': 50.0,
                'rs_momentum': 48.8,
            }
            expected_wr = np.mean([win_rates[s] for s in active])
            
            # Position size (5% base, adjusted by # of strategies)
            position_pct = min(10, 5 * len(active) / 2)
            
            price = df['close'].iloc[-1]
            
            self.signals.append(BlendedSignal(
                ticker=ticker,
                name=self.stock_names.get(ticker, ticker),
                price=round(price, 2),
                cup_handle=signals['cup_handle'],
                volume_dryup=signals['volume_dryup'],
                accumulation=signals['accumulation'],
                flat_base=signals['flat_base'],
                bb_squeeze=signals['bb_squeeze'],
                rs_momentum=signals['rs_momentum'],
                active_strategies=active,
                blended_score=round(normalized_score, 1),
                signal_strength=strength,
                entry_price=round(price, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                risk_reward=round(rr, 2),
                position_size_pct=round(position_pct, 1),
                expected_win_rate=round(expected_wr, 1)
            ))
        
        # Sort by blended score
        self.signals.sort(key=lambda x: x.blended_score, reverse=True)
    
    def run(self) -> str:
        """Run complete scan."""
        self.scan_time = datetime.now()
        self.fetch_data()
        self.scan_all()
        return self.generate_report()
    
    def generate_report(self) -> str:
        """Generate report."""
        lines = []
        
        lines.append("=" * 120)
        lines.append("🏆 OPTIMIZED BLENDED STRATEGY SCANNER")
        lines.append(f"   Generated: {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("   Based on 2019-2026 Backtest (103 stocks, 7+ years)")
        lines.append("=" * 120)
        
        # Strategy Performance Summary
        lines.append(f"""
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PROVEN STRATEGIES (Backtest Results)                                                                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  🥇 Cup & Handle:        PF=1.80, Win=54.2% - U-shaped base + handle + breakout                                        │
│  🥈 Volume Dry-Up:       PF=1.76, Win=54.8% - Low volume consolidation near highs                                      │
│  🥉 Accumulation:        PF=1.74, Win=54.0% - A/D rising + price consolidation                                         │
│     Flat Base:           PF=1.65, Win=53.1% - Tight range (<15%) near highs                                            │
│     BB Squeeze:          PF=1.47, Win=50.0% - Volatility contraction + breakout                                        │
│     RS Momentum:         PF=1.37, Win=48.8% - 60d momentum > 15%, above 50MA                                           │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ❌ DROPPED: VCP Pattern, MA Alignment, Stage 2 Entry (insufficient signals or poor performance)                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
""")
        
        # Active Signals Summary
        strong = [s for s in self.signals if s.signal_strength == "STRONG"]
        moderate = [s for s in self.signals if s.signal_strength == "MODERATE"]
        weak = [s for s in self.signals if s.signal_strength == "WEAK"]
        
        lines.append(f"""
{'='*120}
📊 SCAN SUMMARY
{'='*120}

  Total Stocks Scanned:    {len(self.stock_data)}
  Stocks with Signals:     {len(self.signals)}
  
  Signal Breakdown:
    🔥 STRONG (4+ strategies):   {len(strong)}
    ⚡ MODERATE (2-3 strategies): {len(moderate)}
    📊 WEAK (1 strategy):         {len(weak)}
""")
        
        # Strong Signals
        if strong:
            lines.append(f"""
{'='*120}
🔥 STRONG SIGNALS (4+ Strategies Confirming) - HIGHEST PROBABILITY
{'='*120}

  {'Ticker':<10} {'Company':<30} {'Price':>8} {'Score':>6} {'WinRate':>8} {'Strategies':<40}
  {'-'*115}""")
            
            for s in strong[:10]:
                strats = ', '.join(s.active_strategies)
                lines.append(f"  {s.ticker:<10} {s.name[:30]:<30} {s.price:>8.2f} {s.blended_score:>5.1f} "
                           f"{s.expected_win_rate:>7.1f}% {strats[:40]:<40}")
        
        # Moderate Signals
        if moderate:
            lines.append(f"""

{'='*120}
⚡ MODERATE SIGNALS (2-3 Strategies Confirming)
{'='*120}

  {'Ticker':<10} {'Company':<30} {'Price':>8} {'Score':>6} {'WinRate':>8} {'Strategies':<40}
  {'-'*115}""")
            
            for s in moderate[:15]:
                strats = ', '.join(s.active_strategies)
                lines.append(f"  {s.ticker:<10} {s.name[:30]:<30} {s.price:>8.2f} {s.blended_score:>5.1f} "
                           f"{s.expected_win_rate:>7.1f}% {strats[:40]:<40}")
        
        # Detailed Top 10
        lines.append(f"""

{'='*120}
📋 DETAILED TOP 10 OPPORTUNITIES
{'='*120}
""")
        
        for i, s in enumerate(self.signals[:10]):
            # Strategy icons
            icons = []
            if s.cup_handle:
                icons.append("🏆 Cup&Handle")
            if s.volume_dryup:
                icons.append("📉 VolDryUp")
            if s.accumulation:
                icons.append("💰 Accum")
            if s.flat_base:
                icons.append("📊 FlatBase")
            if s.bb_squeeze:
                icons.append("🔥 BBSqueeze")
            if s.rs_momentum:
                icons.append("⚡ RSMom")
            
            lines.append(f"""
  #{i+1} {s.ticker} - {s.name}
  {'─'*100}
  
  SIGNAL STRENGTH: {s.signal_strength} ({len(s.active_strategies)} strategies confirming)
  BLENDED SCORE:   {s.blended_score:.1f}/100
  EXPECTED WIN:    {s.expected_win_rate:.1f}%
  
  ACTIVE STRATEGIES:
    {' | '.join(icons)}
  
  TRADE PARAMETERS:
    Entry Price:    {s.entry_price:.2f} SAR
    Stop Loss:      {s.stop_loss:.2f} SAR ({(s.stop_loss/s.entry_price-1)*100:.1f}%)
    Target 1:       {s.target_1:.2f} SAR (+15.0%)
    Target 2:       {s.target_2:.2f} SAR (+25.0%)
    Risk/Reward:    {s.risk_reward:.2f}
    Position Size:  {s.position_size_pct:.1f}% of portfolio
""")
        
        # Trading Rules
        lines.append(f"""
{'='*120}
📖 TRADING RULES (Based on Backtest)
{'='*120}

  ENTRY RULES:
    1. Only trade STRONG signals (4+ strategies) or MODERATE (2-3) in BULL market
    2. Entry at current price or on pullback to support
    3. Confirm with volume expansion on breakout day

  EXIT RULES:
    1. Stop Loss: 8% below entry (backtest optimal)
    2. Target 1: +15% (take 50% off)
    3. Target 2: +25% (remaining position)
    4. Time Stop: Exit after 20 days if neither hit

  POSITION SIZING:
    • STRONG signal: 7-10% of portfolio
    • MODERATE signal: 5-7% of portfolio
    • WEAK signal: Skip or 2-3% max
    • Max 5 positions at once
    • Max 30% in any sector

  MARKET FILTER:
    • BULL market (index > 50MA): Full exposure
    • NEUTRAL: 50% exposure, only STRONG signals
    • BEAR market (index < 50MA): 25% exposure or cash

{'='*120}
END OF REPORT
{'='*120}
""")
        
        return "\n".join(lines)
    
    def save_report(self, output_dir: str = "output/production") -> str:
        """Save report."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.scan_time.strftime('%Y%m%d_%H%M%S')
        filepath = f"{output_dir}/optimized_scan_{timestamp}.txt"
        
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        
        with open(f"{output_dir}/latest_optimized_scan.txt", 'w') as f:
            f.write(report)
        
        return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🏆 OPTIMIZED BLENDED STRATEGY SCANNER")
    print("   Using proven strategies from 2019-2026 backtest")
    print("=" * 70 + "\n")
    
    scanner = OptimizedScanner(capital=1_000_000)
    report = scanner.run()
    print(report)
    filepath = scanner.save_report()
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
