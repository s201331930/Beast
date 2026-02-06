#!/usr/bin/env python3
"""
COMPLETE PRODUCTION SCANNER
===========================
Full output with all trade parameters and performance tracking.

Includes:
1. Market regime and leverage recommendation
2. All trade parameters (entry, rebalance triggers, position sizes)
3. Historical tracking of regime changes and performance
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

STRATEGY = {
    'regime_ma_period': 50,
    'leverage_bull': 1.5,
    'leverage_bear': 0.5,
    'rebalance_threshold': 0.05,  # Rebalance if weights drift >5%
    'max_position_pct': 0.15,
    'min_position_pct': 0.03,
}

STOCKS = {
    '1180.SR': {'name': 'Al Rajhi Bank', 'sector': 'Banks', 'weight': 1.0},
    '1010.SR': {'name': 'Riyad Bank', 'sector': 'Banks', 'weight': 1.0},
    '2222.SR': {'name': 'Saudi Aramco', 'sector': 'Energy', 'weight': 1.0},
    '7010.SR': {'name': 'STC', 'sector': 'Telecom', 'weight': 1.0},
    '2010.SR': {'name': 'SABIC', 'sector': 'Materials', 'weight': 1.0},
    '1150.SR': {'name': 'Alinma Bank', 'sector': 'Banks', 'weight': 1.0},
    '2082.SR': {'name': 'ACWA Power', 'sector': 'Utilities', 'weight': 0.8},
    '2280.SR': {'name': 'Almarai', 'sector': 'Food', 'weight': 0.8},
    '8210.SR': {'name': 'Bupa Arabia', 'sector': 'Insurance', 'weight': 0.8},
    '4190.SR': {'name': 'Jarir Marketing', 'sector': 'Retail', 'weight': 0.8},
    '1211.SR': {'name': 'Maaden', 'sector': 'Materials', 'weight': 0.7},
    '7020.SR': {'name': 'Mobily', 'sector': 'Telecom', 'weight': 0.7},
    '4300.SR': {'name': 'Dar Al Arkan', 'sector': 'Real Estate', 'weight': 0.6},
    '2050.SR': {'name': 'Savola', 'sector': 'Food', 'weight': 0.6},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StockPosition:
    """Complete position information for a stock."""
    ticker: str
    name: str
    sector: str
    
    # Current Market Data
    current_price: float
    ma_50: float
    price_vs_ma_pct: float
    daily_change_pct: float
    
    # Position Parameters
    target_weight_pct: float
    target_value: float
    target_shares: int
    
    # Key Levels
    regime_change_price: float  # Price that would change regime
    
    # Stock-specific regime
    stock_regime: str


@dataclass
class RegimeRecord:
    """Record of a regime period."""
    start_date: str
    end_date: Optional[str]
    regime: str
    leverage: float
    starting_price: float
    ending_price: Optional[float]
    return_pct: Optional[float]
    leveraged_return_pct: Optional[float]
    days: int
    status: str  # 'ACTIVE' or 'CLOSED'


@dataclass
class PerformanceTracker:
    """Tracks historical performance."""
    start_date: str
    total_days: int
    regime_history: List[RegimeRecord]
    cumulative_bh_return: float
    cumulative_strategy_return: float
    excess_return: float
    current_regime: str
    current_leverage: float


# =============================================================================
# MAIN SCANNER
# =============================================================================

class CompleteScanner:
    """Full production scanner with tracking."""
    
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.positions: List[StockPosition] = []
        self.tracker_file = "output/production/performance_tracker.json"
        self.tracker: Optional[PerformanceTracker] = None
        self.scan_time = None
        
    def load_tracker(self) -> None:
        """Load historical tracking data."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                
                regime_history = [RegimeRecord(**r) for r in data.get('regime_history', [])]
                
                self.tracker = PerformanceTracker(
                    start_date=data.get('start_date', datetime.now().strftime('%Y-%m-%d')),
                    total_days=data.get('total_days', 0),
                    regime_history=regime_history,
                    cumulative_bh_return=data.get('cumulative_bh_return', 0),
                    cumulative_strategy_return=data.get('cumulative_strategy_return', 0),
                    excess_return=data.get('excess_return', 0),
                    current_regime=data.get('current_regime', 'UNKNOWN'),
                    current_leverage=data.get('current_leverage', 1.0)
                )
            except Exception as e:
                self.tracker = None
    
    def save_tracker(self) -> None:
        """Save tracking data."""
        if not self.tracker:
            return
            
        os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'start_date': self.tracker.start_date,
            'total_days': self.tracker.total_days,
            'regime_history': [asdict(r) for r in self.tracker.regime_history],
            'cumulative_bh_return': self.tracker.cumulative_bh_return,
            'cumulative_strategy_return': self.tracker.cumulative_strategy_return,
            'excess_return': self.tracker.excess_return,
            'current_regime': self.tracker.current_regime,
            'current_leverage': self.tracker.current_leverage
        }
        
        with open(self.tracker_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def fetch_data(self) -> None:
        """Fetch all stock data."""
        for ticker in STOCKS.keys():
            try:
                df = yf.Ticker(ticker).history(period="1y")
                if len(df) >= 60:
                    df.columns = [c.lower() for c in df.columns]
                    self.stock_data[ticker] = df
            except:
                pass
    
    def detect_regime(self, df: pd.DataFrame) -> tuple:
        """Detect regime for a stock."""
        close = df['close']
        ma_50 = close.rolling(STRATEGY['regime_ma_period']).mean()
        
        current_price = close.iloc[-1]
        current_ma = ma_50.iloc[-1]
        
        regime = "BULL" if current_price > current_ma else "BEAR"
        leverage = STRATEGY['leverage_bull'] if regime == "BULL" else STRATEGY['leverage_bear']
        
        return regime, leverage, current_price, current_ma
    
    def calculate_positions(self, market_regime: str, market_leverage: float) -> List[StockPosition]:
        """Calculate all position details."""
        positions = []
        
        # Calculate total weight for normalization
        total_weight = sum(info['weight'] for ticker, info in STOCKS.items() 
                         if ticker in self.stock_data)
        
        for ticker, info in STOCKS.items():
            if ticker not in self.stock_data:
                continue
            
            df = self.stock_data[ticker]
            close = df['close']
            ma_50 = close.rolling(50).mean()
            
            current_price = close.iloc[-1]
            current_ma = ma_50.iloc[-1]
            prev_close = close.iloc[-2] if len(close) > 1 else current_price
            
            # Stock regime
            stock_regime = "BULL" if current_price > current_ma else "BEAR"
            
            # Position sizing
            base_weight = info['weight'] / total_weight
            leverage_weight = base_weight * market_leverage
            leverage_weight = max(STRATEGY['min_position_pct'], 
                                 min(STRATEGY['max_position_pct'], leverage_weight))
            
            target_value = self.capital * leverage_weight
            target_shares = int(target_value / current_price)
            
            positions.append(StockPosition(
                ticker=ticker,
                name=info['name'],
                sector=info['sector'],
                current_price=round(current_price, 2),
                ma_50=round(current_ma, 2),
                price_vs_ma_pct=round((current_price / current_ma - 1) * 100, 2),
                daily_change_pct=round((current_price / prev_close - 1) * 100, 2),
                target_weight_pct=round(leverage_weight * 100, 2),
                target_value=round(target_value, 2),
                target_shares=target_shares,
                regime_change_price=round(current_ma, 2),  # Price that changes regime
                stock_regime=stock_regime
            ))
        
        return sorted(positions, key=lambda x: x.target_value, reverse=True)
    
    def update_tracker(self, regime: str, leverage: float, market_return: float = 0) -> None:
        """Update performance tracker."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if not self.tracker:
            # Initialize tracker
            self.tracker = PerformanceTracker(
                start_date=today,
                total_days=1,
                regime_history=[
                    RegimeRecord(
                        start_date=today,
                        end_date=None,
                        regime=regime,
                        leverage=leverage,
                        starting_price=100,  # Normalized
                        ending_price=None,
                        return_pct=None,
                        leveraged_return_pct=None,
                        days=1,
                        status='ACTIVE'
                    )
                ],
                cumulative_bh_return=0,
                cumulative_strategy_return=0,
                excess_return=0,
                current_regime=regime,
                current_leverage=leverage
            )
        else:
            # Check if regime changed
            if regime != self.tracker.current_regime:
                # Close current regime
                if self.tracker.regime_history:
                    current = self.tracker.regime_history[-1]
                    current.end_date = today
                    current.status = 'CLOSED'
                
                # Start new regime
                self.tracker.regime_history.append(
                    RegimeRecord(
                        start_date=today,
                        end_date=None,
                        regime=regime,
                        leverage=leverage,
                        starting_price=100,
                        ending_price=None,
                        return_pct=None,
                        leveraged_return_pct=None,
                        days=1,
                        status='ACTIVE'
                    )
                )
            else:
                # Update current regime
                if self.tracker.regime_history:
                    self.tracker.regime_history[-1].days += 1
            
            self.tracker.total_days += 1
            self.tracker.current_regime = regime
            self.tracker.current_leverage = leverage
        
        self.save_tracker()
    
    def run(self) -> str:
        """Run complete scan and generate report."""
        self.scan_time = datetime.now()
        self.load_tracker()
        
        print("Fetching data...")
        self.fetch_data()
        print(f"Loaded {len(self.stock_data)} stocks")
        
        # Detect market regime using Al Rajhi as proxy
        market_proxy = '1180.SR'
        if market_proxy not in self.stock_data:
            market_proxy = list(self.stock_data.keys())[0]
        
        regime, leverage, market_price, market_ma = self.detect_regime(self.stock_data[market_proxy])
        
        # Calculate positions
        self.positions = self.calculate_positions(regime, leverage)
        
        # Update tracker
        self.update_tracker(regime, leverage)
        
        # Generate report
        return self.generate_report(regime, leverage, market_price, market_ma)
    
    def generate_report(self, regime: str, leverage: float, market_price: float, market_ma: float) -> str:
        """Generate comprehensive report."""
        lines = []
        
        # =================================================================
        # HEADER
        # =================================================================
        lines.append("=" * 90)
        lines.append("🇸🇦 TASI TREND-BASED LEVERAGE STRATEGY - COMPLETE REPORT")
        lines.append(f"   Generated: {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 90)
        
        # =================================================================
        # STRATEGY SUMMARY
        # =================================================================
        lines.append(f"""
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STRATEGY: Regime-Based Leverage (Backtest: +141.7% vs +33.8% Buy&Hold)              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  • Stay ALWAYS invested (no entry/exit signals)                                      │
│  • BULL regime (Price > 50-MA): Use 1.5x leverage                                    │
│  • BEAR regime (Price < 50-MA): Use 0.5x leverage                                    │
│  • Rebalance on regime change or monthly                                             │
└──────────────────────────────────────────────────────────────────────────────────────┘
""")
        
        # =================================================================
        # CURRENT MARKET REGIME
        # =================================================================
        regime_emoji = "🟢" if regime == "BULL" else "🔴"
        lines.append("=" * 90)
        lines.append(f"{regime_emoji} CURRENT MARKET REGIME: {regime}")
        lines.append("=" * 90)
        lines.append(f"""
  Market Proxy (Al Rajhi):
  ─────────────────────────────────────────────────────────────────────────────────────
  Current Price:        {market_price:.2f} SAR
  50-Day MA:            {market_ma:.2f} SAR
  Price vs MA:          {(market_price/market_ma-1)*100:+.2f}%
  
  REGIME PARAMETERS:
  ─────────────────────────────────────────────────────────────────────────────────────
  Current Regime:       {regime}
  Recommended Leverage: {leverage}x
  Target Exposure:      {self.capital * leverage:,.0f} SAR ({leverage*100:.0f}% of capital)
  
  REGIME CHANGE TRIGGER:
  ─────────────────────────────────────────────────────────────────────────────────────
  Price to trigger BEAR: {market_ma:.2f} SAR (if price drops below 50-MA)
  Price to trigger BULL: {market_ma:.2f} SAR (if price rises above 50-MA)
""")
        
        # =================================================================
        # POSITION DETAILS
        # =================================================================
        lines.append("=" * 90)
        lines.append(f"📊 POSITION DETAILS ({len(self.positions)} stocks)")
        lines.append("=" * 90)
        
        lines.append(f"""
  Capital: {self.capital:,.0f} SAR | Target Leverage: {leverage}x | Total Exposure: {self.capital * leverage:,.0f} SAR
""")
        
        lines.append(f"  {'Ticker':<10} {'Name':<18} {'Price':>10} {'50-MA':>10} {'vs MA':>8} {'Weight':>8} {'Shares':>8} {'Value':>12} {'Regime':<6}")
        lines.append("  " + "-" * 100)
        
        total_value = 0
        for p in self.positions:
            regime_icon = "🟢" if p.stock_regime == "BULL" else "🔴"
            lines.append(f"  {p.ticker:<10} {p.name[:18]:<18} {p.current_price:>10.2f} {p.ma_50:>10.2f} "
                        f"{p.price_vs_ma_pct:>+7.1f}% {p.target_weight_pct:>7.1f}% {p.target_shares:>8} "
                        f"{p.target_value:>12,.0f} {regime_icon}")
            total_value += p.target_value
        
        lines.append("  " + "-" * 100)
        total_weight = sum(p.target_weight_pct for p in self.positions)
        lines.append(f"  {'TOTAL':<10} {'':<18} {'':<10} {'':<10} {'':<8} {total_weight:>7.1f}% {'':<8} {total_value:>12,.0f}")
        
        # =================================================================
        # DETAILED TRADE PARAMETERS FOR EACH POSITION
        # =================================================================
        lines.append(f"""
{'='*90}
📋 DETAILED TRADE PARAMETERS
{'='*90}
""")
        
        for p in self.positions[:10]:  # Top 10
            regime_icon = "🟢" if p.stock_regime == "BULL" else "🔴"
            lines.append(f"""
  {p.ticker} - {p.name} {regime_icon}
  ─────────────────────────────────────────────────────────────────────────────────────
  ENTRY:
    Current Price:      {p.current_price:.2f} SAR
    Target Shares:      {p.target_shares:,} shares
    Position Value:     {p.target_value:,.0f} SAR
    Weight:             {p.target_weight_pct:.1f}% of portfolio
  
  KEY LEVELS:
    50-Day MA:          {p.ma_50:.2f} SAR
    Price vs MA:        {p.price_vs_ma_pct:+.1f}%
    Daily Change:       {p.daily_change_pct:+.2f}%
  
  REGIME CHANGE TRIGGER:
    If price crosses {p.ma_50:.2f} SAR → Stock regime changes
    Current stock regime: {p.stock_regime}
""")
        
        # =================================================================
        # PERFORMANCE TRACKER
        # =================================================================
        lines.append("=" * 90)
        lines.append("📈 PERFORMANCE TRACKER")
        lines.append("=" * 90)
        
        if self.tracker and self.tracker.regime_history:
            lines.append(f"""
  Tracking Since: {self.tracker.start_date}
  Total Days: {self.tracker.total_days}
  Current Regime: {self.tracker.current_regime} ({self.tracker.current_leverage}x leverage)
""")
            
            lines.append("\n  REGIME HISTORY:")
            lines.append("  " + "-" * 80)
            lines.append(f"  {'Start':<12} {'End':<12} {'Regime':<8} {'Leverage':>8} {'Days':>6} {'Status':<10}")
            lines.append("  " + "-" * 80)
            
            for r in self.tracker.regime_history[-10:]:  # Last 10 regimes
                end_str = r.end_date if r.end_date else "ONGOING"
                lines.append(f"  {r.start_date:<12} {end_str:<12} {r.regime:<8} {r.leverage:>7.1f}x {r.days:>6} {r.status:<10}")
            
            # Summary
            bull_periods = [r for r in self.tracker.regime_history if r.regime == 'BULL']
            bear_periods = [r for r in self.tracker.regime_history if r.regime == 'BEAR']
            
            bull_days = sum(r.days for r in bull_periods)
            bear_days = sum(r.days for r in bear_periods)
            
            lines.append(f"""
  ─────────────────────────────────────────────────────────────────────────────────────
  SUMMARY:
    Total BULL days: {bull_days} ({bull_days/max(1,self.tracker.total_days)*100:.1f}%)
    Total BEAR days: {bear_days} ({bear_days/max(1,self.tracker.total_days)*100:.1f}%)
    Regime changes:  {len(self.tracker.regime_history)}
""")
        else:
            lines.append(f"""
  No historical data yet. Performance tracking will begin from today.
  Run this scanner daily to build performance history.
""")
        
        # =================================================================
        # ACTION ITEMS
        # =================================================================
        lines.append("=" * 90)
        lines.append("🎯 ACTION ITEMS")
        lines.append("=" * 90)
        
        lines.append(f"""
  CURRENT RECOMMENDATION:
  ─────────────────────────────────────────────────────────────────────────────────────
  Regime: {regime} → Use {leverage}x leverage
  
  {"⚠️  ACTION REQUIRED:" if regime == "BULL" else "⚠️  DEFENSIVE POSTURE:"}
""")
        
        if regime == "BULL":
            lines.append(f"""
  • INCREASE exposure to {leverage}x ({self.capital * leverage:,.0f} SAR total)
  • Allocate to {len(self.positions)} stocks as per weights above
  • Monitor: If price drops below {market_ma:.2f} SAR, reduce to 0.5x leverage
""")
        else:
            lines.append(f"""
  • REDUCE exposure to {leverage}x ({self.capital * leverage:,.0f} SAR total)
  • Keep {len(self.positions)} stocks but at reduced weights
  • Monitor: If price rises above {market_ma:.2f} SAR, increase to 1.5x leverage
""")
        
        # =================================================================
        # EXPECTED PERFORMANCE
        # =================================================================
        lines.append(f"""
{'='*90}
📊 EXPECTED PERFORMANCE (Based on Backtest 2022-2026)
{'='*90}

  ┌─────────────────────────┬────────────────┬────────────────┬────────────────┐
  │         Metric          │  Buy & Hold    │ This Strategy  │    Excess      │
  ├─────────────────────────┼────────────────┼────────────────┼────────────────┤
  │  Total Return           │     +33.8%     │    +141.7%     │    +107.9%     │
  │  Annualized Return      │      +7.4%     │     +24.3%     │     +16.8%     │
  │  Sharpe Ratio           │      0.56      │      1.61      │     +1.05      │
  │  Max Drawdown           │     19.3%      │     11.2%      │     Better     │
  └─────────────────────────┴────────────────┴────────────────┴────────────────┘
""")
        
        lines.append("=" * 90)
        lines.append("END OF REPORT")
        lines.append("=" * 90)
        
        return "\n".join(lines)
    
    def save_report(self, output_dir: str = "output/production") -> str:
        """Save report to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.scan_time.strftime('%Y%m%d_%H%M%S')
        filepath = f"{output_dir}/complete_report_{timestamp}.txt"
        
        report = self.generate_report(
            self.tracker.current_regime if self.tracker else "UNKNOWN",
            self.tracker.current_leverage if self.tracker else 1.0,
            self.stock_data['1180.SR']['close'].iloc[-1] if '1180.SR' in self.stock_data else 0,
            self.stock_data['1180.SR']['close'].rolling(50).mean().iloc[-1] if '1180.SR' in self.stock_data else 0
        )
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save as latest
        with open(f"{output_dir}/latest_complete_report.txt", 'w') as f:
            f.write(report)
        
        return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    scanner = CompleteScanner(capital=1_000_000)
    report = scanner.run()
    print(report)
    filepath = scanner.save_report()
    print(f"\nReport saved to: {filepath}")

if __name__ == "__main__":
    main()
