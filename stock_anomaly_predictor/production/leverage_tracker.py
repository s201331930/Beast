#!/usr/bin/env python3
"""
LEVERAGE STRATEGY TRACKER
=========================
Tracks the performance of the trend-based leverage strategy over time.
Records regime changes, leverage adjustments, and cumulative returns.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from production.config import STRATEGY, SYSTEM


@dataclass
class DailyRecord:
    """Daily performance record."""
    date: str
    regime: str
    leverage: float
    market_return: float      # Daily market return (equal weight)
    strategy_return: float    # Leveraged return
    cumulative_bh: float      # Cumulative buy & hold
    cumulative_strategy: float  # Cumulative strategy
    excess_return: float      # Strategy - B&H


class LeverageTracker:
    """
    Tracks performance of leverage strategy.
    Persists data for long-term monitoring.
    """
    
    def __init__(self, data_file: str = "output/production/leverage_performance.json"):
        self.data_file = data_file
        self.records: List[DailyRecord] = []
        self.start_capital = 1_000_000
        self.load()
    
    def load(self) -> None:
        """Load existing records."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                self.records = [DailyRecord(**r) for r in data.get('records', [])]
            except:
                pass
    
    def save(self) -> None:
        """Save records to file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'records': [asdict(r) for r in self.records],
            'summary': self.get_summary()
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_record(self, regime: str, leverage: float, market_return: float) -> DailyRecord:
        """Add a new daily record."""
        # Get previous cumulative values
        if self.records:
            prev = self.records[-1]
            cum_bh = prev.cumulative_bh * (1 + market_return)
            cum_strat = prev.cumulative_strategy * (1 + market_return * leverage)
        else:
            cum_bh = 1 + market_return
            cum_strat = 1 + market_return * leverage
        
        strategy_return = market_return * leverage
        excess = cum_strat - cum_bh
        
        record = DailyRecord(
            date=datetime.now().strftime('%Y-%m-%d'),
            regime=regime,
            leverage=leverage,
            market_return=round(market_return, 6),
            strategy_return=round(strategy_return, 6),
            cumulative_bh=round(cum_bh, 6),
            cumulative_strategy=round(cum_strat, 6),
            excess_return=round(excess, 6)
        )
        
        self.records.append(record)
        self.save()
        return record
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        if not self.records:
            return {'days': 0, 'bh_return': 0, 'strategy_return': 0, 'excess': 0}
        
        latest = self.records[-1]
        
        # Calculate metrics
        bh_return = (latest.cumulative_bh - 1) * 100
        strategy_return = (latest.cumulative_strategy - 1) * 100
        excess = strategy_return - bh_return
        
        # Regime breakdown
        bull_days = sum(1 for r in self.records if r.regime == 'BULL')
        bear_days = sum(1 for r in self.records if r.regime == 'BEAR')
        
        # Average leverage
        avg_leverage = np.mean([r.leverage for r in self.records])
        
        return {
            'days': len(self.records),
            'start_date': self.records[0].date,
            'end_date': latest.date,
            'bh_return': round(bh_return, 2),
            'strategy_return': round(strategy_return, 2),
            'excess_return': round(excess, 2),
            'bull_days': bull_days,
            'bear_days': bear_days,
            'avg_leverage': round(avg_leverage, 2),
            'current_regime': latest.regime,
            'current_leverage': latest.leverage
        }
    
    def generate_report(self) -> str:
        """Generate performance report."""
        summary = self.get_summary()
        
        lines = []
        lines.append("=" * 70)
        lines.append("LEVERAGE STRATEGY PERFORMANCE TRACKING")
        lines.append("=" * 70)
        
        if not self.records:
            lines.append("\n  No performance data yet. Run daily scanner to start tracking.")
            return "\n".join(lines)
        
        lines.append(f"""
SUMMARY ({summary['days']} days tracked)
────────────────────────────────────────────────────────────────────────
  Period:           {summary['start_date']} to {summary['end_date']}
  
  Buy & Hold:       {summary['bh_return']:>+8.2f}%
  Strategy:         {summary['strategy_return']:>+8.2f}%
  EXCESS RETURN:    {summary['excess_return']:>+8.2f}%
  
  Bull Days:        {summary['bull_days']} ({summary['bull_days']/summary['days']*100:.1f}%)
  Bear Days:        {summary['bear_days']} ({summary['bear_days']/summary['days']*100:.1f}%)
  Avg Leverage:     {summary['avg_leverage']}x
  
  Current Regime:   {summary['current_regime']}
  Current Leverage: {summary['current_leverage']}x
""")
        
        # Recent records
        lines.append("\nRECENT DAYS (Last 10):")
        lines.append("-" * 70)
        lines.append(f"{'Date':<12} {'Regime':<8} {'Lev':>5} {'Mkt Ret':>10} {'Strat Ret':>10} {'Cum B&H':>10} {'Cum Strat':>10}")
        lines.append("-" * 70)
        
        for r in self.records[-10:]:
            lines.append(f"{r.date:<12} {r.regime:<8} {r.leverage:>5.1f}x "
                        f"{r.market_return*100:>+9.2f}% {r.strategy_return*100:>+9.2f}% "
                        f"{(r.cumulative_bh-1)*100:>+9.2f}% {(r.cumulative_strategy-1)*100:>+9.2f}%")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
