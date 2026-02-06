#!/usr/bin/env python3
"""
PRODUCTION TASI SCANNER
=======================
Scans all Saudi market stocks and identifies trading opportunities
based on the scientifically optimized strategy.

Usage:
    python scanner.py              # Full scan
    python scanner.py --quick      # Quick scan (priority 1 stocks only)
    python scanner.py --email      # Send report via email
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance --quiet")
    import yfinance as yf

from production.config import STRATEGY, TASI_STOCKS, SYSTEM

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StockAnalysis:
    ticker: str
    name: str
    sector: str
    current_price: float
    score: float
    regime: str
    leverage: float
    signal: str
    signal_strength: float
    entry_price: float
    take_profit: float
    trailing_stop: float
    position_size_pct: float
    recommendation: str
    details: Dict


@dataclass
class MarketRegime:
    regime: str
    leverage: float
    ma_50: float
    current_level: float
    volatility: float
    trend_strength: float


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def fetch_stock_data(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch historical data for a stock."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")
        if len(df) < SYSTEM['min_data_days']:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        return None


def calculate_indicators(df: pd.DataFrame) -> Dict:
    """Calculate all technical indicators."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(STRATEGY['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(STRATEGY['rsi_period']).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Moving Averages
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(STRATEGY['regime_ma_period']).mean()
    ma_200 = close.rolling(200).mean()
    
    # Bollinger Bands
    bb_middle = close.rolling(STRATEGY['bb_period']).mean()
    bb_std = close.rolling(STRATEGY['bb_period']).std()
    bb_upper = bb_middle + STRATEGY['bb_std'] * bb_std
    bb_lower = bb_middle - STRATEGY['bb_std'] * bb_std
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Volume
    volume_ma = volume.rolling(20).mean()
    volume_ratio = volume / volume_ma
    
    # Momentum
    momentum = close / close.shift(STRATEGY['momentum_period']) - 1
    
    # Volatility
    returns = close.pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    
    # ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # Trend Strength
    trend_strength = (close - ma_50) / ma_50
    
    return {
        'rsi': rsi,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'ma_200': ma_200,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_position': bb_position,
        'volume_ratio': volume_ratio,
        'momentum': momentum,
        'volatility': volatility,
        'atr': atr,
        'trend_strength': trend_strength,
        'returns': returns,
    }


def detect_market_regime(indicators: Dict, current_price: float) -> MarketRegime:
    """Detect current market regime."""
    ma_50 = indicators['ma_50'].iloc[-1]
    volatility = indicators['volatility'].iloc[-1]
    trend_strength = indicators['trend_strength'].iloc[-1]
    
    # Base regime from MA
    if current_price > ma_50:
        regime = "BULL"
        base_leverage = STRATEGY['leverage_bull']
    else:
        regime = "BEAR"
        base_leverage = STRATEGY['leverage_bear']
    
    # Adjust for volatility
    if volatility > STRATEGY['regime_vol_threshold']:
        leverage = base_leverage * STRATEGY['leverage_high_vol_multiplier']
        regime = f"{regime}_HIGH_VOL"
    else:
        leverage = base_leverage
    
    return MarketRegime(
        regime=regime,
        leverage=round(leverage, 2),
        ma_50=round(ma_50, 2),
        current_level=round(current_price, 2),
        volatility=round(volatility, 4),
        trend_strength=round(trend_strength, 4)
    )


def calculate_score(indicators: Dict, current_price: float) -> Tuple[float, Dict]:
    """Calculate stock score (0-100)."""
    weights = STRATEGY['score_weights']
    scores = {}
    
    # Momentum Score (higher momentum = higher score)
    mom = indicators['momentum'].iloc[-1]
    scores['momentum'] = min(100, max(0, 50 + mom * 500))  # Scale to 0-100
    
    # Volatility Score (lower vol = higher score)
    vol = indicators['volatility'].iloc[-1]
    scores['volatility'] = min(100, max(0, 100 - vol * 200))
    
    # Volume Score (higher volume = higher score)
    vol_ratio = indicators['volume_ratio'].iloc[-1]
    scores['volume'] = min(100, max(0, vol_ratio * 40))
    
    # RSI Score (oversold = higher score for entry)
    rsi = indicators['rsi'].iloc[-1]
    if rsi < 30:
        scores['rsi'] = 90
    elif rsi < 40:
        scores['rsi'] = 70
    elif rsi < 50:
        scores['rsi'] = 50
    elif rsi < 60:
        scores['rsi'] = 40
    else:
        scores['rsi'] = 20
    
    # Trend Score (above MA = higher score)
    ma_50 = indicators['ma_50'].iloc[-1]
    if current_price > ma_50:
        scores['trend'] = 70 + min(30, (current_price / ma_50 - 1) * 300)
    else:
        scores['trend'] = 30 + max(0, (current_price / ma_50) * 30)
    
    # Beta Score (moderate beta = higher score)
    # Approximated from volatility
    vol = indicators['volatility'].iloc[-1]
    if 0.15 < vol < 0.30:
        scores['beta'] = 80
    elif 0.10 < vol < 0.40:
        scores['beta'] = 60
    else:
        scores['beta'] = 40
    
    # Weighted total
    total = sum(scores[k] * weights[k] for k in weights.keys())
    
    return round(total, 1), {k: round(v, 1) for k, v in scores.items()}


def detect_signal(indicators: Dict, current_price: float) -> Tuple[str, float]:
    """Detect entry signal and strength."""
    rsi = indicators['rsi'].iloc[-1]
    bb_pos = indicators['bb_position'].iloc[-1]
    vol_ratio = indicators['volume_ratio'].iloc[-1]
    momentum = indicators['momentum'].iloc[-1]
    ma_50 = indicators['ma_50'].iloc[-1]
    
    signal = "NONE"
    strength = 0.0
    
    # Mean Reversion Signal
    mr_conditions = [
        rsi < STRATEGY['rsi_oversold'],
        bb_pos < 0.3,
        vol_ratio > STRATEGY['volume_ratio_threshold']
    ]
    mr_score = sum(mr_conditions) / len(mr_conditions)
    
    # Momentum Signal
    mom_conditions = [
        momentum > STRATEGY['momentum_threshold'],
        current_price > ma_50,
        rsi < 60  # Not overbought
    ]
    mom_score = sum(mom_conditions) / len(mom_conditions)
    
    # Determine signal
    if mr_score >= 0.66:
        signal = "OVERSOLD_ENTRY"
        strength = mr_score
    elif mom_score >= 0.66:
        signal = "MOMENTUM_ENTRY"
        strength = mom_score
    elif mr_score >= 0.5 or mom_score >= 0.5:
        signal = "WATCH"
        strength = max(mr_score, mom_score)
    
    return signal, round(strength, 2)


def analyze_stock(ticker: str, info: Dict) -> Optional[StockAnalysis]:
    """Perform full analysis on a single stock."""
    df = fetch_stock_data(ticker)
    if df is None:
        return None
    
    current_price = df['close'].iloc[-1]
    indicators = calculate_indicators(df)
    regime = detect_market_regime(indicators, current_price)
    score, score_details = calculate_score(indicators, current_price)
    signal, signal_strength = detect_signal(indicators, current_price)
    
    # Calculate trade parameters
    atr = indicators['atr'].iloc[-1]
    entry_price = current_price * 1.002  # With slippage
    take_profit = entry_price * (1 + STRATEGY['take_profit_pct'])
    trailing_stop = entry_price * (1 - STRATEGY['trailing_stop_pct'])
    
    # Position size based on score and regime
    base_size = STRATEGY['max_position_pct']
    if score >= 80:
        size_multiplier = 1.0
    elif score >= 70:
        size_multiplier = 0.8
    elif score >= 60:
        size_multiplier = 0.6
    else:
        size_multiplier = 0.4
    
    position_size = base_size * size_multiplier * regime.leverage
    position_size = max(STRATEGY['min_position_pct'], 
                       min(STRATEGY['max_position_pct'], position_size))
    
    # Recommendation
    if score >= STRATEGY['min_score'] and signal in ["OVERSOLD_ENTRY", "MOMENTUM_ENTRY"]:
        recommendation = "BUY"
    elif score >= STRATEGY['min_score'] and signal == "WATCH":
        recommendation = "WATCHLIST"
    elif score < 50:
        recommendation = "AVOID"
    else:
        recommendation = "HOLD"
    
    return StockAnalysis(
        ticker=ticker,
        name=info['name'],
        sector=info['sector'],
        current_price=round(current_price, 2),
        score=score,
        regime=regime.regime,
        leverage=regime.leverage,
        signal=signal,
        signal_strength=signal_strength,
        entry_price=round(entry_price, 2),
        take_profit=round(take_profit, 2),
        trailing_stop=round(trailing_stop, 2),
        position_size_pct=round(position_size * 100, 1),
        recommendation=recommendation,
        details={
            'score_breakdown': score_details,
            'rsi': round(indicators['rsi'].iloc[-1], 1),
            'momentum': round(indicators['momentum'].iloc[-1] * 100, 2),
            'volatility': round(indicators['volatility'].iloc[-1] * 100, 2),
            'volume_ratio': round(indicators['volume_ratio'].iloc[-1], 2),
            'ma_50': round(indicators['ma_50'].iloc[-1], 2),
            'trend_strength': round(regime.trend_strength * 100, 2),
        }
    )


# =============================================================================
# SCANNER CLASS
# =============================================================================

class TASIScanner:
    """Production scanner for TASI stocks."""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results: List[StockAnalysis] = []
        self.market_regime: Optional[MarketRegime] = None
        self.scan_time: Optional[datetime] = None
    
    def run_scan(self) -> None:
        """Run full market scan."""
        self.scan_time = datetime.now()
        print("=" * 80)
        print(f"TASI MARKET SCANNER - {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Filter stocks based on mode
        if self.quick_mode:
            stocks = {k: v for k, v in TASI_STOCKS.items() if v['priority'] == 1}
            print(f"Quick mode: Scanning {len(stocks)} priority stocks")
        else:
            stocks = TASI_STOCKS
            print(f"Full mode: Scanning {len(stocks)} stocks")
        
        print("\n[1] FETCHING DATA AND ANALYZING...")
        print("-" * 60)
        
        # Detect overall market regime using market proxy (Al Rajhi)
        market_df = fetch_stock_data('1180.SR')
        if market_df is not None:
            market_indicators = calculate_indicators(market_df)
            self.market_regime = detect_market_regime(
                market_indicators, 
                market_df['close'].iloc[-1]
            )
            print(f"\n  MARKET REGIME: {self.market_regime.regime}")
            print(f"  Recommended Leverage: {self.market_regime.leverage}x")
            print(f"  Market Volatility: {self.market_regime.volatility*100:.1f}%")
        
        # Scan all stocks
        self.results = []
        for i, (ticker, info) in enumerate(stocks.items()):
            try:
                result = analyze_stock(ticker, info)
                if result:
                    self.results.append(result)
                    status = "✓" if result.recommendation == "BUY" else "·"
                    print(f"  {status} {ticker}: {result.name[:20]:<20} Score:{result.score:>5.1f} Signal:{result.signal:<15} -> {result.recommendation}")
            except Exception as e:
                print(f"  ✗ {ticker}: Error - {str(e)[:30]}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(stocks)} processed")
        
        print(f"\n  Completed: {len(self.results)} stocks analyzed")
    
    def get_buy_signals(self) -> List[StockAnalysis]:
        """Get stocks with BUY recommendation."""
        return [r for r in self.results if r.recommendation == "BUY"]
    
    def get_watchlist(self) -> List[StockAnalysis]:
        """Get stocks on watchlist."""
        return [r for r in self.results if r.recommendation == "WATCHLIST"]
    
    def get_top_scores(self, n: int = 10) -> List[StockAnalysis]:
        """Get top N stocks by score."""
        return sorted(self.results, key=lambda x: x.score, reverse=True)[:n]
    
    def generate_report(self) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"TASI SCANNER REPORT - {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        # Market regime
        if self.market_regime:
            lines.append(f"\nMARKET REGIME: {self.market_regime.regime}")
            lines.append(f"Leverage Setting: {self.market_regime.leverage}x")
            lines.append(f"Volatility: {self.market_regime.volatility*100:.1f}%")
        
        # Buy signals
        buy_signals = self.get_buy_signals()
        lines.append(f"\n{'='*80}")
        lines.append(f"BUY SIGNALS ({len(buy_signals)} stocks)")
        lines.append("=" * 80)
        
        if buy_signals:
            lines.append(f"\n{'Ticker':<10} {'Name':<20} {'Price':>10} {'Score':>8} {'Signal':>15} {'Size%':>8}")
            lines.append("-" * 80)
            for s in sorted(buy_signals, key=lambda x: x.score, reverse=True):
                lines.append(f"{s.ticker:<10} {s.name[:20]:<20} {s.current_price:>10.2f} {s.score:>8.1f} {s.signal:>15} {s.position_size_pct:>7.1f}%")
            
            # Trade details
            lines.append(f"\nTRADE PARAMETERS:")
            lines.append("-" * 80)
            for s in sorted(buy_signals, key=lambda x: x.score, reverse=True):
                lines.append(f"\n{s.ticker} - {s.name}")
                lines.append(f"  Entry Price:    {s.entry_price:.2f} SAR")
                lines.append(f"  Take Profit:    {s.take_profit:.2f} SAR (+{STRATEGY['take_profit_pct']*100:.0f}%)")
                lines.append(f"  Trailing Stop:  {s.trailing_stop:.2f} SAR (-{STRATEGY['trailing_stop_pct']*100:.0f}%)")
                lines.append(f"  Position Size:  {s.position_size_pct:.1f}% of capital")
                lines.append(f"  RSI: {s.details['rsi']:.1f}, Momentum: {s.details['momentum']:.1f}%, Vol: {s.details['volatility']:.1f}%")
        else:
            lines.append("\n  No active buy signals at this time.")
        
        # Watchlist
        watchlist = self.get_watchlist()
        lines.append(f"\n{'='*80}")
        lines.append(f"WATCHLIST ({len(watchlist)} stocks)")
        lines.append("=" * 80)
        
        if watchlist:
            for s in sorted(watchlist, key=lambda x: x.score, reverse=True)[:10]:
                lines.append(f"  {s.ticker:<10} {s.name[:20]:<20} Score:{s.score:>5.1f} RSI:{s.details['rsi']:>5.1f}")
        
        # Top scores
        lines.append(f"\n{'='*80}")
        lines.append("TOP 10 BY SCORE")
        lines.append("=" * 80)
        for s in self.get_top_scores(10):
            lines.append(f"  {s.ticker:<10} {s.name[:20]:<20} Score:{s.score:>5.1f} {s.recommendation:<10}")
        
        # Strategy reminder
        lines.append(f"\n{'='*80}")
        lines.append("STRATEGY PARAMETERS")
        lines.append("=" * 80)
        lines.append(f"  Regime Detection:    50-day MA")
        lines.append(f"  Bull Leverage:       {STRATEGY['leverage_bull']}x")
        lines.append(f"  Bear Leverage:       {STRATEGY['leverage_bear']}x")
        lines.append(f"  Take Profit:         {STRATEGY['take_profit_pct']*100:.0f}%")
        lines.append(f"  Trailing Stop:       {STRATEGY['trailing_stop_pct']*100:.0f}% (activates at +{STRATEGY['trailing_activation_pct']*100:.0f}%)")
        lines.append(f"  Max Holding:         {STRATEGY['max_holding_days']} days")
        lines.append(f"  Min Score:           {STRATEGY['min_score']}")
        
        lines.append(f"\n{'='*80}")
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: str = None) -> None:
        """Save results to files."""
        if output_dir is None:
            output_dir = SYSTEM['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.scan_time.strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_data = {
            'scan_time': self.scan_time.isoformat(),
            'market_regime': asdict(self.market_regime) if self.market_regime else None,
            'strategy_params': STRATEGY,
            'results': [asdict(r) for r in self.results],
            'buy_signals': [asdict(r) for r in self.get_buy_signals()],
            'watchlist': [asdict(r) for r in self.get_watchlist()],
        }
        
        with open(f"{output_dir}/scan_{timestamp}.json", 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Save CSV
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(f"{output_dir}/scan_{timestamp}.csv", index=False)
        
        # Save report
        report = self.generate_report()
        with open(f"{output_dir}/report_{timestamp}.txt", 'w') as f:
            f.write(report)
        
        # Save latest (for easy access)
        with open(f"{output_dir}/latest.json", 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        with open(f"{output_dir}/latest_report.txt", 'w') as f:
            f.write(report)
        
        print(f"\n  Results saved to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TASI Market Scanner')
    parser.add_argument('--quick', action='store_true', help='Quick scan (priority 1 only)')
    parser.add_argument('--email', action='store_true', help='Send report via email')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Run scanner
    scanner = TASIScanner(quick_mode=args.quick)
    scanner.run_scan()
    
    # Generate and print report
    report = scanner.generate_report()
    print(report)
    
    # Save results
    scanner.save_results(args.output)
    
    # Email if requested
    if args.email:
        try:
            from production.emailer import send_report
            send_report(report, scanner.get_buy_signals())
            print("\n  Email sent successfully!")
        except Exception as e:
            print(f"\n  Email failed: {e}")


if __name__ == "__main__":
    main()
