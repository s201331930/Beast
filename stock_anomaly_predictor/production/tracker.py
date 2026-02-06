#!/usr/bin/env python3
"""
TRADE TRACKER
=============
Tracks all flagged trades and monitors their performance.
Maintains history of signals and their outcomes.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

from production.config import STRATEGY


@dataclass
class TrackedTrade:
    """A trade being tracked from signal to exit."""
    trade_id: str
    ticker: str
    name: str
    sector: str
    
    # Entry details
    signal_date: str
    signal_type: str
    signal_strength: float
    regime: str
    leverage: float
    score: float
    
    # Price levels
    entry_price: float
    take_profit: float
    trailing_stop: float
    current_stop: float
    
    # Position
    position_size_pct: float
    
    # Tracking
    status: str  # 'ACTIVE', 'CLOSED_TP', 'CLOSED_TRAIL', 'CLOSED_TIME', 'CLOSED_MANUAL'
    high_price: float
    current_price: float
    current_return_pct: float
    days_held: int
    
    # History
    price_history: List[Dict] = field(default_factory=list)
    
    # Exit details (if closed)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    final_return_pct: Optional[float] = None


class TradeTracker:
    """
    Manages tracking of all flagged trades.
    Persists data to JSON for continuity across runs.
    """
    
    def __init__(self, data_file: str = "output/production/tracked_trades.json"):
        self.data_file = data_file
        self.trades: Dict[str, TrackedTrade] = {}
        self.closed_trades: List[TrackedTrade] = []
        self.load()
    
    def load(self) -> None:
        """Load existing trades from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load active trades
                for trade_id, trade_data in data.get('active_trades', {}).items():
                    self.trades[trade_id] = TrackedTrade(**trade_data)
                
                # Load closed trades
                for trade_data in data.get('closed_trades', []):
                    self.closed_trades.append(TrackedTrade(**trade_data))
                    
            except Exception as e:
                print(f"Error loading trades: {e}")
    
    def save(self) -> None:
        """Save trades to file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        data = {
            'last_updated': datetime.now().isoformat(),
            'active_trades': {tid: asdict(t) for tid, t in self.trades.items()},
            'closed_trades': [asdict(t) for t in self.closed_trades],
            'summary': self.get_summary()
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add_trade(self, signal_data: Dict) -> TrackedTrade:
        """Add a new trade from scanner signal."""
        trade_id = f"{signal_data['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = TrackedTrade(
            trade_id=trade_id,
            ticker=signal_data['ticker'],
            name=signal_data['name'],
            sector=signal_data['sector'],
            signal_date=datetime.now().strftime('%Y-%m-%d'),
            signal_type=signal_data['signal'],
            signal_strength=signal_data['signal_strength'],
            regime=signal_data['regime'],
            leverage=signal_data['leverage'],
            score=signal_data['score'],
            entry_price=signal_data['entry_price'],
            take_profit=signal_data['take_profit'],
            trailing_stop=signal_data['trailing_stop'],
            current_stop=signal_data['trailing_stop'],
            position_size_pct=signal_data['position_size_pct'],
            status='ACTIVE',
            high_price=signal_data['current_price'],
            current_price=signal_data['current_price'],
            current_return_pct=0.0,
            days_held=0,
            price_history=[{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'price': signal_data['current_price'],
                'return_pct': 0.0
            }]
        )
        
        self.trades[trade_id] = trade
        self.save()
        return trade
    
    def update_trade(self, trade_id: str, current_price: float) -> TrackedTrade:
        """Update a trade with current price."""
        if trade_id not in self.trades:
            raise ValueError(f"Trade {trade_id} not found")
        
        trade = self.trades[trade_id]
        trade.current_price = current_price
        trade.current_return_pct = (current_price / trade.entry_price - 1) * 100
        trade.days_held = (datetime.now() - datetime.strptime(trade.signal_date, '%Y-%m-%d')).days
        
        # Update high water mark
        if current_price > trade.high_price:
            trade.high_price = current_price
            # Update trailing stop if in profit
            if trade.current_return_pct >= STRATEGY['trailing_activation_pct'] * 100:
                new_stop = trade.high_price * (1 - STRATEGY['trailing_stop_pct'])
                trade.current_stop = max(trade.current_stop, new_stop)
        
        # Add to price history
        trade.price_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'price': current_price,
            'return_pct': trade.current_return_pct
        })
        
        # Check exit conditions
        exit_reason = None
        
        # Take profit
        if current_price >= trade.take_profit:
            exit_reason = 'TAKE_PROFIT'
        # Trailing stop
        elif current_price <= trade.current_stop and trade.current_return_pct > 0:
            exit_reason = 'TRAILING_STOP'
        # Time exit
        elif trade.days_held >= STRATEGY['max_holding_days']:
            exit_reason = 'TIME_EXIT'
        
        if exit_reason:
            self.close_trade(trade_id, current_price, exit_reason)
        else:
            self.save()
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> TrackedTrade:
        """Close a trade."""
        if trade_id not in self.trades:
            raise ValueError(f"Trade {trade_id} not found")
        
        trade = self.trades[trade_id]
        trade.status = f'CLOSED_{exit_reason}'
        trade.exit_date = datetime.now().strftime('%Y-%m-%d')
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.final_return_pct = (exit_price / trade.entry_price - 1) * 100
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.trades[trade_id]
        
        self.save()
        return trade
    
    def update_all_trades(self, current_prices: Dict[str, float]) -> List[TrackedTrade]:
        """Update all active trades with current prices."""
        updated = []
        for trade_id in list(self.trades.keys()):
            trade = self.trades[trade_id]
            if trade.ticker in current_prices:
                self.update_trade(trade_id, current_prices[trade.ticker])
                updated.append(trade)
        return updated
    
    def get_active_trades(self) -> List[TrackedTrade]:
        """Get all active trades."""
        return list(self.trades.values())
    
    def get_closed_trades(self, days: int = None) -> List[TrackedTrade]:
        """Get closed trades, optionally filtered by recent days."""
        if days is None:
            return self.closed_trades
        
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in self.closed_trades 
                if datetime.strptime(t.exit_date, '%Y-%m-%d') >= cutoff]
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        active = self.get_active_trades()
        closed = self.closed_trades
        
        if not closed:
            return {
                'active_trades': len(active),
                'closed_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }
        
        wins = [t for t in closed if t.final_return_pct > 0]
        losses = [t for t in closed if t.final_return_pct <= 0]
        
        win_rate = len(wins) / len(closed) if closed else 0
        avg_return = np.mean([t.final_return_pct for t in closed]) if closed else 0
        total_return = sum(t.final_return_pct for t in closed)
        
        # By exit reason
        by_reason = {}
        for t in closed:
            reason = t.exit_reason
            if reason not in by_reason:
                by_reason[reason] = {'count': 0, 'total_return': 0}
            by_reason[reason]['count'] += 1
            by_reason[reason]['total_return'] += t.final_return_pct
        
        return {
            'active_trades': len(active),
            'closed_trades': len(closed),
            'win_rate': round(win_rate * 100, 1),
            'avg_return': round(avg_return, 2),
            'total_return': round(total_return, 2),
            'avg_win': round(np.mean([t.final_return_pct for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t.final_return_pct for t in losses]), 2) if losses else 0,
            'by_exit_reason': by_reason,
            'current_exposure': sum(t.position_size_pct for t in active)
        }
    
    def generate_report(self) -> str:
        """Generate text report of all tracked trades."""
        lines = []
        lines.append("=" * 80)
        lines.append("TRADE TRACKING REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        # Summary
        summary = self.get_summary()
        lines.append(f"""
PERFORMANCE SUMMARY
───────────────────────────────────────────────────────────────────────────────
  Active Trades:     {summary['active_trades']}
  Closed Trades:     {summary['closed_trades']}
  Win Rate:          {summary['win_rate']}%
  Avg Return:        {summary['avg_return']:+.2f}%
  Total Return:      {summary['total_return']:+.2f}%
  Current Exposure:  {summary.get('current_exposure', 0):.1f}%
""")
        
        # Active trades
        active = self.get_active_trades()
        lines.append("=" * 80)
        lines.append(f"ACTIVE TRADES ({len(active)})")
        lines.append("=" * 80)
        
        if active:
            lines.append(f"\n{'Ticker':<10} {'Entry':>10} {'Current':>10} {'Return':>10} {'Days':>6} {'T/P':>10} {'Stop':>10} {'Status':<15}")
            lines.append("-" * 95)
            
            for t in sorted(active, key=lambda x: x.current_return_pct, reverse=True):
                status = "WINNING" if t.current_return_pct > 0 else "LOSING"
                lines.append(f"{t.ticker:<10} {t.entry_price:>10.2f} {t.current_price:>10.2f} "
                           f"{t.current_return_pct:>+9.2f}% {t.days_held:>6} "
                           f"{t.take_profit:>10.2f} {t.current_stop:>10.2f} {status:<15}")
        else:
            lines.append("\n  No active trades")
        
        # Recent closed trades
        recent_closed = self.get_closed_trades(days=30)
        lines.append(f"\n{'='*80}")
        lines.append(f"RECENTLY CLOSED TRADES (Last 30 days) - {len(recent_closed)} trades")
        lines.append("=" * 80)
        
        if recent_closed:
            lines.append(f"\n{'Ticker':<10} {'Entry':>10} {'Exit':>10} {'Return':>10} {'Days':>6} {'Reason':<15}")
            lines.append("-" * 75)
            
            for t in sorted(recent_closed, key=lambda x: x.exit_date, reverse=True):
                lines.append(f"{t.ticker:<10} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
                           f"{t.final_return_pct:>+9.2f}% {t.days_held:>6} {t.exit_reason:<15}")
        else:
            lines.append("\n  No recently closed trades")
        
        # Exit reason analysis
        if summary.get('by_exit_reason'):
            lines.append(f"\n{'='*80}")
            lines.append("EXIT REASON ANALYSIS")
            lines.append("=" * 80)
            lines.append(f"\n{'Reason':<20} {'Count':>10} {'Avg Return':>12}")
            lines.append("-" * 45)
            for reason, stats in summary['by_exit_reason'].items():
                avg = stats['total_return'] / stats['count'] if stats['count'] > 0 else 0
                lines.append(f"{reason:<20} {stats['count']:>10} {avg:>+11.2f}%")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
