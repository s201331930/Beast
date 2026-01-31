#!/usr/bin/env python3
"""
FAST PARAMETER OPTIMIZATION - Focused ranges
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from itertools import product
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 80)
print("PARAMETER OPTIMIZATION - TASI Trading Strategy")
print("=" * 80)

# Focused parameter ranges
PARAM_GRID = {
    'stop_loss_atr': [2.0, 2.5, 3.0, 4.0, 5.0],
    'take_profit_pct': [0.10, 0.15, 0.20, 0.25, 0.30],
    'max_holding_days': [10, 20, 30, 40, 60],
    'trailing_stop_pct': [0.03, 0.05, 0.07, 0.10],
}

total = len(PARAM_GRID['stop_loss_atr']) * len(PARAM_GRID['take_profit_pct']) * \
        len(PARAM_GRID['max_holding_days']) * len(PARAM_GRID['trailing_stop_pct'])

print(f"\nParameter Ranges:")
for k, v in PARAM_GRID.items():
    print(f"  {k}: {v}")
print(f"\nTotal Combinations: {total}")

# Stocks to test
STOCKS = ['1180.SR', '7010.SR', '7020.SR', '1150.SR', '4300.SR', 
          '1320.SR', '1211.SR', '3020.SR', '1010.SR', '4190.SR']

print(f"\n[1] FETCHING DATA...")
stock_data = {}
for ticker in STOCKS:
    try:
        df = yf.Ticker(ticker).history(start="2022-01-01")
        if len(df) >= 100:
            df.columns = [c.lower() for c in df.columns]
            stock_data[ticker] = df
    except:
        pass
print(f"  Loaded {len(stock_data)} stocks")

# Get trading dates
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index.tolist())
trading_dates = sorted(all_dates)
print(f"  Trading days: {len(trading_dates)}")

# Pre-calculate indicators
print("\n[2] CALCULATING INDICATORS...")
indicators = {}
signals = {}

for ticker, df in stock_data.items():
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # ATR
    if 'high' in df.columns:
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - close.shift(1)),
            abs(df['low'] - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    else:
        atr = close.rolling(14).std()
    
    # Bollinger Band position
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_pos = (close - (sma - 2*std)) / (4*std)
    
    # Volume ratio
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean() if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    
    indicators[ticker] = {
        'rsi': rsi,
        'atr': atr,
        'bb_pos': bb_pos,
        'vol_ratio': vol_ratio
    }
    
    # Signal: RSI < 35, BB < 0.3, Volume > 1.5x
    signals[ticker] = ((rsi < 35) & (bb_pos < 0.3) & (vol_ratio > 1.5)).astype(int)

print("  Done")

# Run optimization
print(f"\n[3] RUNNING {total} SIMULATIONS...")

results = []
combinations = list(product(
    PARAM_GRID['stop_loss_atr'],
    PARAM_GRID['take_profit_pct'],
    PARAM_GRID['max_holding_days'],
    PARAM_GRID['trailing_stop_pct']
))

for idx, (sl_atr, tp_pct, max_days, trail_pct) in enumerate(combinations):
    if (idx + 1) % 50 == 0:
        print(f"  Progress: {idx+1}/{total}")
    
    # Run simulation
    cash = 1_000_000
    positions = {}
    trades = []
    equity = []
    
    for i, date in enumerate(trading_dates[:-1]):
        next_date = trading_dates[i + 1]
        
        # Get prices
        current_prices = {t: df.loc[date, 'close'] for t, df in stock_data.items() if date in df.index}
        next_opens = {t: df.loc[next_date, 'open'] if 'open' in df.columns else df.loc[next_date, 'close']
                     for t, df in stock_data.items() if next_date in df.index}
        
        # Check exits
        for ticker in list(positions.keys()):
            if ticker not in current_prices:
                continue
            
            pos = positions[ticker]
            price = current_prices[ticker]
            holding = (date - pos['entry_date']).days
            
            # Update trailing stop
            if price > pos['high']:
                pos['high'] = price
                pos['trail'] = max(pos['trail'], price * (1 - trail_pct))
            
            stop = max(pos['stop'], pos['trail'])
            
            exit_reason = None
            if price <= stop:
                exit_reason = 'stop_loss'
            elif price >= pos['tp']:
                exit_reason = 'take_profit'
            elif holding >= max_days:
                exit_reason = 'time_exit'
            
            if exit_reason and ticker in next_opens:
                exit_price = next_opens[ticker] * 0.999
                pnl = (exit_price - pos['entry']) * pos['shares'] - exit_price * pos['shares'] * 0.001
                
                trades.append({
                    'pnl': pnl,
                    'pnl_pct': exit_price / pos['entry'] - 1,
                    'holding': holding,
                    'reason': exit_reason
                })
                
                cash += exit_price * pos['shares'] * 0.999
                del positions[ticker]
        
        # Check entries
        for ticker in stock_data.keys():
            if ticker in positions or len(positions) >= 15:
                continue
            if date not in signals[ticker].index:
                continue
            if signals[ticker].loc[date] != 1:
                continue
            if ticker not in next_opens:
                continue
            
            entry = next_opens[ticker] * 1.001
            atr_val = indicators[ticker]['atr'].loc[date] if date in indicators[ticker]['atr'].index else entry * 0.02
            if pd.isna(atr_val):
                atr_val = entry * 0.02
            
            shares = int(cash * 0.05 / entry)
            if shares <= 0 or entry * shares > cash * 0.95:
                continue
            
            positions[ticker] = {
                'entry_date': date,
                'entry': entry,
                'shares': shares,
                'stop': entry - atr_val * sl_atr,
                'tp': entry * (1 + tp_pct),
                'trail': entry - atr_val * sl_atr,
                'high': entry
            }
            cash -= entry * shares * 1.001
        
        # Record equity
        pos_val = sum(pos['shares'] * current_prices.get(t, pos['entry']) for t, pos in positions.items())
        equity.append(cash + pos_val)
    
    # Calculate metrics
    if len(trades) < 10:
        continue
    
    final = cash + sum(pos['shares'] * pos['entry'] for pos in positions.values())
    total_return = final / 1_000_000 - 1
    
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(winning) / len(trades)
    profit_factor = sum(t['pnl'] for t in winning) / abs(sum(t['pnl'] for t in losing)) if losing else 999
    
    if len(equity) > 1:
        eq = np.array(equity)
        rets = np.diff(eq) / eq[:-1]
        sharpe = np.sqrt(252) * np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
        peak = np.maximum.accumulate(eq)
        max_dd = ((peak - eq) / peak).max()
    else:
        sharpe = 0
        max_dd = 0
    
    stop_rate = len([t for t in trades if t['reason'] == 'stop_loss']) / len(trades)
    tp_rate = len([t for t in trades if t['reason'] == 'take_profit']) / len(trades)
    time_rate = len([t for t in trades if t['reason'] == 'time_exit']) / len(trades)
    
    results.append({
        'sl_atr': sl_atr,
        'tp_pct': tp_pct,
        'max_days': max_days,
        'trail_pct': trail_pct,
        'return': total_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'pf': profit_factor,
        'max_dd': max_dd,
        'trades': len(trades),
        'stop_rate': stop_rate,
        'tp_rate': tp_rate,
        'time_rate': time_rate,
        'avg_hold': np.mean([t['holding'] for t in trades])
    })

results_df = pd.DataFrame(results)
print(f"\n  Completed {len(results_df)} valid configurations")

# Analysis
print("\n" + "=" * 80)
print("TOP 10 BY TOTAL RETURN")
print("=" * 80)
top = results_df.nlargest(10, 'return')
print(f"\n{'SL':>6} {'TP%':>6} {'Days':>6} {'Trail':>6} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'SL%':>6} {'TP%':>6}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{r['sl_atr']:>6.1f} {r['tp_pct']*100:>5.0f}% {r['max_days']:>6.0f} {r['trail_pct']*100:>5.0f}% | "
          f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} {r['stop_rate']*100:>5.1f}% {r['tp_rate']*100:>5.1f}%")

print("\n" + "=" * 80)
print("TOP 10 BY SHARPE RATIO")
print("=" * 80)
top = results_df.nlargest(10, 'sharpe')
print(f"\n{'SL':>6} {'TP%':>6} {'Days':>6} {'Trail':>6} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'SL%':>6} {'TP%':>6}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{r['sl_atr']:>6.1f} {r['tp_pct']*100:>5.0f}% {r['max_days']:>6.0f} {r['trail_pct']*100:>5.0f}% | "
          f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} {r['stop_rate']*100:>5.1f}% {r['tp_rate']*100:>5.1f}%")

print("\n" + "=" * 80)
print("TOP 10 BY PROFIT FACTOR")
print("=" * 80)
top = results_df[results_df['pf'] < 100].nlargest(10, 'pf')
print(f"\n{'SL':>6} {'TP%':>6} {'Days':>6} {'Trail':>6} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'SL%':>6} {'TP%':>6}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{r['sl_atr']:>6.1f} {r['tp_pct']*100:>5.0f}% {r['max_days']:>6.0f} {r['trail_pct']*100:>5.0f}% | "
          f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} {r['stop_rate']*100:>5.1f}% {r['tp_rate']*100:>5.1f}%")

# Parameter sensitivity
print("\n" + "=" * 80)
print("PARAMETER SENSITIVITY ANALYSIS")
print("=" * 80)

print("\n1. STOP-LOSS ATR MULTIPLIER:")
by_sl = results_df.groupby('sl_atr').agg({'return': 'mean', 'win_rate': 'mean', 'stop_rate': 'mean', 'pf': 'mean'}).round(4)
print(by_sl.to_string())

print("\n2. TAKE-PROFIT PERCENTAGE:")
by_tp = results_df.groupby('tp_pct').agg({'return': 'mean', 'win_rate': 'mean', 'tp_rate': 'mean', 'pf': 'mean'}).round(4)
by_tp.index = [f"{x*100:.0f}%" for x in by_tp.index]
print(by_tp.to_string())

print("\n3. MAX HOLDING DAYS:")
by_hold = results_df.groupby('max_days').agg({'return': 'mean', 'win_rate': 'mean', 'time_rate': 'mean', 'pf': 'mean'}).round(4)
print(by_hold.to_string())

print("\n4. TRAILING STOP %:")
by_trail = results_df.groupby('trail_pct').agg({'return': 'mean', 'win_rate': 'mean', 'pf': 'mean'}).round(4)
by_trail.index = [f"{x*100:.0f}%" for x in by_trail.index]
print(by_trail.to_string())

# Best configuration
print("\n" + "=" * 80)
print("OPTIMAL CONFIGURATION")
print("=" * 80)

# Score: weighted combination
results_df['score'] = (
    results_df['return'] * 0.4 +
    results_df['sharpe'].clip(-2, 2) / 4 * 0.3 +
    results_df['win_rate'] * 0.15 +
    results_df['pf'].clip(0, 3) / 3 * 0.15
)

best = results_df.loc[results_df['score'].idxmax()]

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDED OPTIMAL PARAMETERS                          │
├────────────────────────────────────────────────────────────────────────────────┤
│   Stop-Loss ATR:       {best['sl_atr']:>6.1f}x                                              │
│   Take-Profit:         {best['tp_pct']*100:>5.0f}%                                               │
│   Max Holding Days:    {best['max_days']:>5.0f}                                                │
│   Trailing Stop:       {best['trail_pct']*100:>5.0f}%                                               │
├────────────────────────────────────────────────────────────────────────────────┤
│   EXPECTED PERFORMANCE:                                                        │
│   Total Return:        {best['return']*100:>+6.1f}%                                              │
│   Sharpe Ratio:        {best['sharpe']:>+6.2f}                                               │
│   Win Rate:            {best['win_rate']*100:>5.1f}%                                              │
│   Profit Factor:       {best['pf']:>6.2f}                                               │
│   Max Drawdown:        {best['max_dd']*100:>5.1f}%                                              │
│   Total Trades:        {best['trades']:>5.0f}                                                │
│   Avg Holding:         {best['avg_hold']:>5.1f} days                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│   EXIT BREAKDOWN:                                                              │
│   Stop-Loss:           {best['stop_rate']*100:>5.1f}%                                              │
│   Take-Profit:         {best['tp_rate']*100:>5.1f}%                                              │
│   Time Exit:           {best['time_rate']*100:>5.1f}%                                              │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Compare with original
print("\n" + "=" * 80)
print("COMPARISON: ORIGINAL vs OPTIMAL")
print("=" * 80)
print(f"""
┌─────────────────────┬────────────┬────────────┬────────────┐
│     Parameter       │  Original  │  Optimal   │   Change   │
├─────────────────────┼────────────┼────────────┼────────────┤
│ Stop-Loss ATR       │    2.0x    │    {best['sl_atr']:.1f}x    │   {best['sl_atr']-2.0:+.1f}x    │
│ Take-Profit %       │    10%     │    {best['tp_pct']*100:.0f}%     │   {(best['tp_pct']-0.10)*100:+.0f}%    │
│ Max Holding Days    │    20      │    {best['max_days']:.0f}      │   {best['max_days']-20:+.0f}      │
│ Trailing Stop %     │    3%      │    {best['trail_pct']*100:.0f}%      │   {(best['trail_pct']-0.03)*100:+.0f}%     │
├─────────────────────┼────────────┼────────────┼────────────┤
│ Expected Return     │   -6.9%    │  {best['return']*100:>+5.1f}%   │  {(best['return']+0.069)*100:>+5.1f}%   │
│ Win Rate            │   41.0%    │  {best['win_rate']*100:>5.1f}%   │  {(best['win_rate']-0.41)*100:>+5.1f}%   │
│ Stop-Loss Rate      │   71.4%    │  {best['stop_rate']*100:>5.1f}%   │  {(best['stop_rate']-0.714)*100:>+5.1f}%   │
└─────────────────────┴────────────┴────────────┴────────────┘
""")

# Save results
os.makedirs('output/optimization', exist_ok=True)
results_df.to_csv('output/optimization/all_results.csv', index=False)

optimal = {
    'stop_loss_atr': float(best['sl_atr']),
    'take_profit_pct': float(best['tp_pct']),
    'max_holding_days': int(best['max_days']),
    'trailing_stop_pct': float(best['trail_pct']),
    'expected_return': float(best['return']),
    'expected_sharpe': float(best['sharpe']),
    'expected_win_rate': float(best['win_rate']),
    'expected_profit_factor': float(best['pf'])
}

with open('output/optimization/optimal_config.json', 'w') as f:
    json.dump(optimal, f, indent=2)

print(f"\nResults saved to output/optimization/")
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
