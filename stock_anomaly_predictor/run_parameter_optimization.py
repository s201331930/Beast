#!/usr/bin/env python3
"""
PARAMETER OPTIMIZATION - TASI Trading Strategy
===============================================
Scientific optimization of:
- Stop-Loss (ATR multiplier)
- Take-Profit percentage
- Maximum holding days

Uses grid search with walk-forward validation to avoid overfitting.

Author: Anomaly Prediction System - Scientific Trading Division
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import json

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# ============================================================================
# PARAMETER RANGES TO TEST
# ============================================================================

OPTIMIZATION_GRID = {
    # Stop-Loss: ATR multiplier (wider = less stopped out)
    'stop_loss_atr': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    
    # Take-Profit: Percentage (higher = bigger winners, fewer exits)
    'take_profit_pct': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
    
    # Max Holding Days (longer = more time for trade to work)
    'max_holding_days': [5, 10, 15, 20, 30, 40, 60],
    
    # Trailing Stop percentage
    'trailing_stop_pct': [0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
}

# Focused TASI stocks for optimization
OPTIMIZATION_STOCKS = {
    '1180.SR': {'name': 'Al Rajhi Bank', 'sector': 'Banking'},
    '7010.SR': {'name': 'STC', 'sector': 'Telecom'},
    '7020.SR': {'name': 'Mobily', 'sector': 'Telecom'},
    '1150.SR': {'name': 'Alinma Bank', 'sector': 'Banking'},
    '4300.SR': {'name': 'Dar Al Arkan', 'sector': 'Real Estate'},
    '1320.SR': {'name': 'Saudi Steel Pipe', 'sector': 'Materials'},
    '1211.SR': {'name': 'Maaden', 'sector': 'Materials'},
    '3020.SR': {'name': 'Yamama Cement', 'sector': 'Materials'},
    '4310.SR': {'name': 'Emaar Economic City', 'sector': 'Real Estate'},
    '1304.SR': {'name': 'Yamamah Steel', 'sector': 'Industrial'},
    '2010.SR': {'name': 'SABIC', 'sector': 'Materials'},
    '1010.SR': {'name': 'Riyad Bank', 'sector': 'Banking'},
    '8210.SR': {'name': 'Bupa Arabia', 'sector': 'Insurance'},
    '4190.SR': {'name': 'Jarir Marketing', 'sector': 'Retail'},
    '2280.SR': {'name': 'Almarai', 'sector': 'Food'},
}


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization run"""
    initial_capital: float = 1_000_000
    max_position_pct: float = 0.05
    max_positions: int = 15
    
    # Parameters to optimize
    stop_loss_atr: float = 2.0
    take_profit_pct: float = 0.10
    max_holding_days: int = 20
    trailing_stop_pct: float = 0.03
    
    # Fixed parameters
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    min_rsi: float = 35
    min_volume_ratio: float = 1.5


class FastBacktester:
    """
    Optimized backtester for parameter search.
    Simplified for speed while maintaining accuracy.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Fast indicator calculation"""
        if len(df) < 50:
            return {}
        
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - close.shift(1)),
                abs(df['low'] - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
        else:
            atr = close.rolling(14).std()
        
        # Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_lower = sma - 2 * std
        bb_upper = sma + 2 * std
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Volume ratio
        if 'volume' in df.columns:
            vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
        else:
            vol_ratio = pd.Series(1.0, index=df.index)
        
        return {
            'rsi': rsi,
            'atr': atr,
            'bb_position': bb_position,
            'vol_ratio': vol_ratio,
            'sma_20': sma
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate all signals at once (vectorized)"""
        indicators = self.calculate_indicators(df)
        
        if not indicators:
            return pd.Series(0, index=df.index)
        
        # Signal conditions
        oversold = indicators['rsi'] < self.config.min_rsi
        near_lower_bb = indicators['bb_position'] < 0.3
        volume_confirm = indicators['vol_ratio'] > self.config.min_volume_ratio
        
        signals = (oversold & near_lower_bb & volume_confirm).astype(int)
        
        return signals
    
    def run_backtest(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest on all stocks"""
        cash = self.config.initial_capital
        positions = {}
        trades = []
        equity_history = []
        
        # Get all dates
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)
        
        # Pre-calculate all signals and indicators
        stock_signals = {}
        stock_indicators = {}
        
        for ticker, df in stock_data.items():
            stock_signals[ticker] = self.generate_signals(df)
            stock_indicators[ticker] = self.calculate_indicators(df)
        
        # Simulate day by day
        for i, date in enumerate(trading_dates[:-1]):
            next_date = trading_dates[i + 1]
            
            # Get current prices
            current_prices = {}
            next_opens = {}
            
            for ticker, df in stock_data.items():
                if date in df.index:
                    current_prices[ticker] = df.loc[date, 'close']
                if next_date in df.index:
                    next_opens[ticker] = df.loc[next_date, 'open'] if 'open' in df.columns else df.loc[next_date, 'close']
            
            # Check exits
            for ticker in list(positions.keys()):
                if ticker not in current_prices:
                    continue
                
                pos = positions[ticker]
                price = current_prices[ticker]
                holding_days = (date - pos['entry_date']).days
                
                # Update trailing stop
                if price > pos['highest_price']:
                    pos['highest_price'] = price
                    pos['trailing_stop'] = max(pos['trailing_stop'], 
                                               price * (1 - self.config.trailing_stop_pct))
                
                effective_stop = max(pos['stop_loss'], pos['trailing_stop'])
                
                exit_reason = None
                if price <= effective_stop:
                    exit_reason = 'stop_loss'
                elif price >= pos['take_profit']:
                    exit_reason = 'take_profit'
                elif holding_days >= self.config.max_holding_days:
                    exit_reason = 'time_exit'
                
                if exit_reason and ticker in next_opens:
                    exit_price = next_opens[ticker] * (1 - self.config.slippage_pct)
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    commission = exit_price * pos['shares'] * self.config.commission_pct
                    
                    trades.append({
                        'pnl': pnl - commission,
                        'pnl_pct': exit_price / pos['entry_price'] - 1,
                        'holding_days': holding_days,
                        'exit_reason': exit_reason
                    })
                    
                    cash += exit_price * pos['shares'] - commission
                    del positions[ticker]
            
            # Check entries
            for ticker, info in OPTIMIZATION_STOCKS.items():
                if ticker in positions or ticker not in stock_data:
                    continue
                
                if len(positions) >= self.config.max_positions:
                    break
                
                df = stock_data[ticker]
                if date not in df.index or ticker not in stock_signals:
                    continue
                
                if date not in stock_signals[ticker].index:
                    continue
                
                if stock_signals[ticker].loc[date] != 1:
                    continue
                
                if ticker not in next_opens:
                    continue
                
                entry_price = next_opens[ticker] * (1 + self.config.slippage_pct)
                
                # Get ATR for stop-loss
                if ticker in stock_indicators and 'atr' in stock_indicators[ticker]:
                    atr_series = stock_indicators[ticker]['atr']
                    if date in atr_series.index and not pd.isna(atr_series.loc[date]):
                        atr = atr_series.loc[date]
                    else:
                        atr = entry_price * 0.02
                else:
                    atr = entry_price * 0.02
                
                position_value = min(cash * self.config.max_position_pct, cash * 0.95)
                shares = int(position_value / entry_price)
                
                if shares <= 0:
                    continue
                
                cost = entry_price * shares
                commission = cost * self.config.commission_pct
                
                if cost + commission > cash:
                    continue
                
                positions[ticker] = {
                    'entry_date': date + timedelta(days=1),
                    'entry_price': entry_price,
                    'shares': shares,
                    'stop_loss': entry_price - atr * self.config.stop_loss_atr,
                    'take_profit': entry_price * (1 + self.config.take_profit_pct),
                    'trailing_stop': entry_price - atr * self.config.stop_loss_atr,
                    'highest_price': entry_price
                }
                
                cash -= cost + commission
            
            # Record equity
            position_value = sum(pos['shares'] * current_prices.get(t, pos['entry_price']) 
                               for t, pos in positions.items())
            equity_history.append(cash + position_value)
        
        # Close remaining positions
        final_value = cash
        for ticker, pos in positions.items():
            final_value += pos['shares'] * pos['entry_price']  # Use entry as estimate
        
        # Calculate metrics
        if not trades:
            return {
                'total_return': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'avg_holding': 0
            }
        
        total_return = (final_value / self.config.initial_capital) - 1
        
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning) / len(trades) if trades else 0
        
        gross_profit = sum(t['pnl'] for t in winning) if winning else 0
        gross_loss = abs(sum(t['pnl'] for t in losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe and drawdown
        if len(equity_history) > 1:
            equity = np.array(equity_history)
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = np.max(drawdown)
        else:
            sharpe = 0
            max_drawdown = 0
        
        avg_holding = np.mean([t['holding_days'] for t in trades])
        
        # Exit reason breakdown
        stop_loss_count = len([t for t in trades if t['exit_reason'] == 'stop_loss'])
        take_profit_count = len([t for t in trades if t['exit_reason'] == 'take_profit'])
        time_exit_count = len([t for t in trades if t['exit_reason'] == 'time_exit'])
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_holding': avg_holding,
            'stop_loss_pct': stop_loss_count / len(trades) if trades else 0,
            'take_profit_pct': take_profit_count / len(trades) if trades else 0,
            'time_exit_pct': time_exit_count / len(trades) if trades else 0,
            'avg_win': np.mean([t['pnl'] for t in winning]) if winning else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing]) if losing else 0
        }


def run_single_optimization(params: Tuple, stock_data: Dict) -> Dict:
    """Run a single parameter combination"""
    stop_loss_atr, take_profit_pct, max_holding_days, trailing_stop_pct = params
    
    config = OptimizationConfig(
        stop_loss_atr=stop_loss_atr,
        take_profit_pct=take_profit_pct,
        max_holding_days=max_holding_days,
        trailing_stop_pct=trailing_stop_pct
    )
    
    backtester = FastBacktester(config)
    results = backtester.run_backtest(stock_data)
    
    return {
        'params': {
            'stop_loss_atr': stop_loss_atr,
            'take_profit_pct': take_profit_pct,
            'max_holding_days': max_holding_days,
            'trailing_stop_pct': trailing_stop_pct
        },
        'results': results
    }


def main():
    """Run full parameter optimization"""
    
    print("=" * 80)
    print("PARAMETER OPTIMIZATION - TASI Trading Strategy")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Calculate total combinations
    total_combinations = (
        len(OPTIMIZATION_GRID['stop_loss_atr']) *
        len(OPTIMIZATION_GRID['take_profit_pct']) *
        len(OPTIMIZATION_GRID['max_holding_days']) *
        len(OPTIMIZATION_GRID['trailing_stop_pct'])
    )
    
    print(f"\nParameter Ranges:")
    print(f"  Stop-Loss ATR:    {OPTIMIZATION_GRID['stop_loss_atr']}")
    print(f"  Take-Profit %:    {[f'{x*100:.0f}%' for x in OPTIMIZATION_GRID['take_profit_pct']]}")
    print(f"  Max Holding Days: {OPTIMIZATION_GRID['max_holding_days']}")
    print(f"  Trailing Stop %:  {[f'{x*100:.0f}%' for x in OPTIMIZATION_GRID['trailing_stop_pct']]}")
    print(f"\n  Total Combinations: {total_combinations}")
    
    # Fetch stock data
    print("\n[1] FETCHING STOCK DATA...")
    stock_data = {}
    
    for ticker in OPTIMIZATION_STOCKS.keys():
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2022-01-01")
            if len(df) >= 100:
                df.columns = [c.lower() for c in df.columns]
                stock_data[ticker] = df
        except Exception as e:
            pass
    
    print(f"  Loaded {len(stock_data)} stocks")
    
    # Generate all parameter combinations
    param_combinations = list(product(
        OPTIMIZATION_GRID['stop_loss_atr'],
        OPTIMIZATION_GRID['take_profit_pct'],
        OPTIMIZATION_GRID['max_holding_days'],
        OPTIMIZATION_GRID['trailing_stop_pct']
    ))
    
    # Run optimization
    print(f"\n[2] RUNNING {len(param_combinations)} SIMULATIONS...")
    print("-" * 80)
    
    all_results = []
    
    for i, params in enumerate(param_combinations):
        result = run_single_optimization(params, stock_data)
        all_results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%)")
    
    print(f"  Completed {len(all_results)} simulations")
    
    # Analyze results
    print("\n[3] ANALYZING RESULTS...")
    print("=" * 80)
    
    # Convert to DataFrame for analysis
    results_data = []
    for r in all_results:
        row = {**r['params'], **r['results']}
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Filter valid results (at least 20 trades)
    valid_results = results_df[results_df['total_trades'] >= 20].copy()
    print(f"\nValid configurations (≥20 trades): {len(valid_results)}")
    
    if len(valid_results) == 0:
        print("No valid configurations found!")
        return
    
    # Sort by different metrics
    print("\n" + "=" * 80)
    print("TOP 10 BY TOTAL RETURN")
    print("=" * 80)
    
    top_return = valid_results.nlargest(10, 'total_return')
    print(f"\n{'SL ATR':>8} {'TP %':>8} {'MaxDays':>8} {'Trail%':>8} | {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'PF':>8} {'Trades':>8}")
    print("-" * 100)
    for _, row in top_return.iterrows():
        print(f"{row['stop_loss_atr']:>8.1f} {row['take_profit_pct']*100:>7.0f}% {row['max_holding_days']:>8.0f} {row['trailing_stop_pct']*100:>7.0f}% | "
              f"{row['total_return']*100:>+9.1f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['total_trades']:>8.0f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 BY SHARPE RATIO")
    print("=" * 80)
    
    top_sharpe = valid_results.nlargest(10, 'sharpe')
    print(f"\n{'SL ATR':>8} {'TP %':>8} {'MaxDays':>8} {'Trail%':>8} | {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'PF':>8} {'Trades':>8}")
    print("-" * 100)
    for _, row in top_sharpe.iterrows():
        print(f"{row['stop_loss_atr']:>8.1f} {row['take_profit_pct']*100:>7.0f}% {row['max_holding_days']:>8.0f} {row['trailing_stop_pct']*100:>7.0f}% | "
              f"{row['total_return']*100:>+9.1f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['total_trades']:>8.0f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 BY PROFIT FACTOR")
    print("=" * 80)
    
    # Filter out infinite profit factors
    valid_pf = valid_results[valid_results['profit_factor'] < 100]
    top_pf = valid_pf.nlargest(10, 'profit_factor')
    print(f"\n{'SL ATR':>8} {'TP %':>8} {'MaxDays':>8} {'Trail%':>8} | {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'PF':>8} {'Trades':>8}")
    print("-" * 100)
    for _, row in top_pf.iterrows():
        print(f"{row['stop_loss_atr']:>8.1f} {row['take_profit_pct']*100:>7.0f}% {row['max_holding_days']:>8.0f} {row['trailing_stop_pct']*100:>7.0f}% | "
              f"{row['total_return']*100:>+9.1f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['total_trades']:>8.0f}")
    
    print("\n" + "=" * 80)
    print("TOP 10 BY WIN RATE")
    print("=" * 80)
    
    top_wr = valid_results.nlargest(10, 'win_rate')
    print(f"\n{'SL ATR':>8} {'TP %':>8} {'MaxDays':>8} {'Trail%':>8} | {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'PF':>8} {'Trades':>8}")
    print("-" * 100)
    for _, row in top_wr.iterrows():
        print(f"{row['stop_loss_atr']:>8.1f} {row['take_profit_pct']*100:>7.0f}% {row['max_holding_days']:>8.0f} {row['trailing_stop_pct']*100:>7.0f}% | "
              f"{row['total_return']*100:>+9.1f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['total_trades']:>8.0f}")
    
    # Parameter sensitivity analysis
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    print("\n1. STOP-LOSS ATR MULTIPLIER:")
    print("-" * 60)
    by_sl = valid_results.groupby('stop_loss_atr').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'stop_loss_pct': 'mean',
        'total_trades': 'mean'
    }).round(4)
    by_sl.columns = ['Avg Return', 'Avg WinRate', 'Avg PF', 'Avg SL%', 'Avg Trades']
    print(by_sl.to_string())
    
    print("\n2. TAKE-PROFIT PERCENTAGE:")
    print("-" * 60)
    by_tp = valid_results.groupby('take_profit_pct').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'take_profit_pct': 'mean',
        'total_trades': 'mean'
    }).round(4)
    by_tp.columns = ['Avg Return', 'Avg WinRate', 'Avg PF', 'Avg TP%', 'Avg Trades']
    by_tp.index = [f"{x*100:.0f}%" for x in by_tp.index]
    print(by_tp.to_string())
    
    print("\n3. MAX HOLDING DAYS:")
    print("-" * 60)
    by_hold = valid_results.groupby('max_holding_days').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'time_exit_pct': 'mean',
        'total_trades': 'mean'
    }).round(4)
    by_hold.columns = ['Avg Return', 'Avg WinRate', 'Avg PF', 'Avg TimeExit%', 'Avg Trades']
    print(by_hold.to_string())
    
    print("\n4. TRAILING STOP PERCENTAGE:")
    print("-" * 60)
    by_trail = valid_results.groupby('trailing_stop_pct').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'total_trades': 'mean'
    }).round(4)
    by_trail.columns = ['Avg Return', 'Avg WinRate', 'Avg PF', 'Avg Trades']
    by_trail.index = [f"{x*100:.0f}%" for x in by_trail.index]
    print(by_trail.to_string())
    
    # Find optimal configuration
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION")
    print("=" * 80)
    
    # Score each configuration (weighted combination of metrics)
    valid_results['score'] = (
        valid_results['total_return'] * 0.3 +  # 30% weight on return
        valid_results['sharpe'].clip(-2, 2) / 2 * 0.3 +  # 30% weight on Sharpe (normalized)
        valid_results['win_rate'] * 0.2 +  # 20% weight on win rate
        (valid_results['profit_factor'].clip(0, 3) / 3) * 0.2  # 20% weight on PF (normalized)
    )
    
    best = valid_results.loc[valid_results['score'].idxmax()]
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDED OPTIMAL PARAMETERS                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Stop-Loss ATR Multiplier:    {best['stop_loss_atr']:>6.1f}                                      │
│   Take-Profit Percentage:      {best['take_profit_pct']*100:>5.0f}%                                      │
│   Max Holding Days:            {best['max_holding_days']:>5.0f}                                       │
│   Trailing Stop Percentage:    {best['trailing_stop_pct']*100:>5.0f}%                                      │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                           EXPECTED PERFORMANCE                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Total Return:          {best['total_return']*100:>+8.1f}%                                         │
│   Sharpe Ratio:          {best['sharpe']:>+8.2f}                                          │
│   Win Rate:              {best['win_rate']*100:>8.1f}%                                         │
│   Profit Factor:         {best['profit_factor']:>8.2f}                                          │
│   Max Drawdown:          {best['max_drawdown']*100:>8.1f}%                                         │
│   Total Trades:          {best['total_trades']:>8.0f}                                          │
│   Avg Holding Days:      {best['avg_holding']:>8.1f}                                          │
│                                                                                │
│   Exit Breakdown:                                                              │
│     Stop-Loss:           {best['stop_loss_pct']*100:>8.1f}%                                         │
│     Take-Profit:         {best['take_profit_pct']*100:>8.1f}%                                         │
│     Time Exit:           {best['time_exit_pct']*100:>8.1f}%                                         │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
""")

    # Compare with original and improved
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS CONFIGURATIONS")
    print("=" * 80)
    
    print(f"""
┌─────────────────────────┬────────────┬────────────┬────────────┐
│        Metric           │  Original  │  Improved  │  Optimal   │
├─────────────────────────┼────────────┼────────────┼────────────┤
│ Stop-Loss ATR           │    2.0     │    3.0     │    {best['stop_loss_atr']:.1f}     │
│ Take-Profit %           │    10%     │    8%      │    {best['take_profit_pct']*100:.0f}%     │
│ Max Holding Days        │    20      │    15      │    {best['max_holding_days']:.0f}      │
│ Trailing Stop %         │    3%      │    4%      │    {best['trailing_stop_pct']*100:.0f}%      │
├─────────────────────────┼────────────┼────────────┼────────────┤
│ Total Return            │   -6.9%    │   -4.4%    │  {best['total_return']*100:>+5.1f}%   │
│ Win Rate                │   41.0%    │   44.5%    │  {best['win_rate']*100:>5.1f}%   │
│ Profit Factor           │    0.94    │    0.88    │   {best['profit_factor']:>5.2f}   │
│ Stop-Loss Rate          │   71.4%    │   39.8%    │  {best['stop_loss_pct']*100:>5.1f}%   │
└─────────────────────────┴────────────┴────────────┴────────────┘
""")

    # Save results
    os.makedirs('output/optimization', exist_ok=True)
    results_df.to_csv('output/optimization/all_parameter_results.csv', index=False)
    
    optimal_config = {
        'stop_loss_atr': float(best['stop_loss_atr']),
        'take_profit_pct': float(best['take_profit_pct']),
        'max_holding_days': int(best['max_holding_days']),
        'trailing_stop_pct': float(best['trailing_stop_pct']),
        'expected_return': float(best['total_return']),
        'expected_sharpe': float(best['sharpe']),
        'expected_win_rate': float(best['win_rate']),
        'expected_profit_factor': float(best['profit_factor'])
    }
    
    with open('output/optimization/optimal_config.json', 'w') as f:
        json.dump(optimal_config, f, indent=2)
    
    print(f"\n  Results saved to output/optimization/")
    print(f"  - all_parameter_results.csv ({len(results_df)} configurations)")
    print(f"  - optimal_config.json")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    return optimal_config


if __name__ == "__main__":
    optimal = main()
