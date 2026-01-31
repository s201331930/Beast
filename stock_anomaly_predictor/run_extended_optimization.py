#!/usr/bin/env python3
"""
EXTENDED PARAMETER OPTIMIZATION
==============================
Testing more extreme parameters based on findings:
- Take-profit targets up to 50%
- Holding periods up to 90 days
- Wider stop-losses (or no stop)
"""

import os
import warnings
from datetime import datetime, timedelta
from itertools import product

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 80)
print("EXTENDED PARAMETER OPTIMIZATION - Wider Ranges")
print("=" * 80)

# Extended parameter ranges based on insights
PARAMS = {
    'stop_loss_atr': [3.0, 4.0, 5.0, 7.0, 10.0, 999],  # 999 = no stop loss
    'take_profit_pct': [0.20, 0.30, 0.40, 0.50, 999],   # 999 = no take profit
    'max_holding_days': [40, 60, 90, 120, 180],
    'trailing_stop_pct': [0.05, 0.07, 0.10, 0.15, 999], # 999 = no trailing
}

total = len(PARAMS['stop_loss_atr']) * len(PARAMS['take_profit_pct']) * \
        len(PARAMS['max_holding_days']) * len(PARAMS['trailing_stop_pct'])
print(f"\nTotal Combinations: {total}")

STOCKS = ['1180.SR', '7010.SR', '7020.SR', '1150.SR', '4300.SR', 
          '1320.SR', '1211.SR', '3020.SR', '1010.SR', '4190.SR',
          '2010.SR', '1080.SR', '2280.SR', '8210.SR', '2050.SR']

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
print(f"  Loaded: {len(stock_data)} stocks")

# Get dates
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index.tolist())
trading_dates = sorted(all_dates)

# Pre-calculate indicators
print("\n[2] CALCULATING INDICATORS...")
indicators = {}
signals = {}

for ticker, df in stock_data.items():
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    if 'high' in df.columns:
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - close.shift(1)),
            abs(df['low'] - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    else:
        atr = close.rolling(14).std()
    
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_pos = (close - (sma - 2*std)) / (4*std)
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean() if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    
    indicators[ticker] = {'rsi': rsi, 'atr': atr, 'bb_pos': bb_pos}
    signals[ticker] = ((rsi < 35) & (bb_pos < 0.3) & (vol_ratio > 1.5)).astype(int)

print("  Done")

# Run optimization
print(f"\n[3] RUNNING {total} SIMULATIONS...")

results = []
combinations = list(product(
    PARAMS['stop_loss_atr'],
    PARAMS['take_profit_pct'],
    PARAMS['max_holding_days'],
    PARAMS['trailing_stop_pct']
))

for idx, (sl_atr, tp_pct, max_days, trail_pct) in enumerate(combinations):
    if (idx + 1) % 100 == 0:
        print(f"  Progress: {idx+1}/{total}")
    
    cash = 1_000_000
    positions = {}
    trades = []
    equity = []
    
    for i, date in enumerate(trading_dates[:-1]):
        next_date = trading_dates[i + 1]
        
        current_prices = {t: df.loc[date, 'close'] for t, df in stock_data.items() if date in df.index}
        next_opens = {t: df.loc[next_date, 'open'] if 'open' in df.columns else df.loc[next_date, 'close']
                     for t, df in stock_data.items() if next_date in df.index}
        
        for ticker in list(positions.keys()):
            if ticker not in current_prices:
                continue
            
            pos = positions[ticker]
            price = current_prices[ticker]
            holding = (date - pos['entry_date']).days
            
            # Trailing stop (if enabled)
            if trail_pct < 100 and price > pos['high']:
                pos['high'] = price
                pos['trail'] = max(pos['trail'], price * (1 - trail_pct))
            
            # Effective stop
            if sl_atr < 100:
                stop = max(pos['stop'], pos.get('trail', 0))
            else:
                stop = 0  # No stop loss
            
            exit_reason = None
            if stop > 0 and price <= stop:
                exit_reason = 'stop_loss'
            elif tp_pct < 100 and price >= pos['tp']:
                exit_reason = 'take_profit'
            elif holding >= max_days:
                exit_reason = 'time_exit'
            
            if exit_reason and ticker in next_opens:
                exit_price = next_opens[ticker] * 0.999
                pnl = (exit_price - pos['entry']) * pos['shares']
                
                trades.append({
                    'pnl': pnl * 0.998,  # commissions
                    'pnl_pct': exit_price / pos['entry'] - 1,
                    'holding': holding,
                    'reason': exit_reason
                })
                
                cash += exit_price * pos['shares'] * 0.999
                del positions[ticker]
        
        # Entries
        for ticker in stock_data.keys():
            if ticker in positions or len(positions) >= 15:
                continue
            if date not in signals[ticker].index or signals[ticker].loc[date] != 1:
                continue
            if ticker not in next_opens:
                continue
            
            entry = next_opens[ticker] * 1.001
            atr_val = indicators[ticker]['atr'].loc[date] if date in indicators[ticker]['atr'].index else entry * 0.02
            if pd.isna(atr_val):
                atr_val = entry * 0.02
            
            shares = int(cash * 0.05 / entry)
            if shares <= 0 or entry * shares > cash * 0.9:
                continue
            
            positions[ticker] = {
                'entry_date': date,
                'entry': entry,
                'shares': shares,
                'stop': entry - atr_val * sl_atr if sl_atr < 100 else 0,
                'tp': entry * (1 + tp_pct) if tp_pct < 100 else 9999999,
                'trail': entry - atr_val * sl_atr if sl_atr < 100 and trail_pct < 100 else 0,
                'high': entry
            }
            cash -= entry * shares * 1.001
        
        pos_val = sum(pos['shares'] * current_prices.get(t, pos['entry']) for t, pos in positions.items())
        equity.append(cash + pos_val)
    
    if len(trades) < 10:
        continue
    
    final = cash + sum(pos['shares'] * pos['entry'] for pos in positions.values())
    total_ret = final / 1_000_000 - 1
    
    winning = [t for t in trades if t['pnl'] > 0]
    losing = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(winning) / len(trades)
    pf = sum(t['pnl'] for t in winning) / abs(sum(t['pnl'] for t in losing)) if losing else 999
    
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
        'return': total_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'pf': pf,
        'max_dd': max_dd,
        'trades': len(trades),
        'stop_rate': stop_rate,
        'tp_rate': tp_rate,
        'time_rate': time_rate,
        'avg_hold': np.mean([t['holding'] for t in trades])
    })

results_df = pd.DataFrame(results)
print(f"\n  Completed {len(results_df)} valid configurations")

# Format helper
def format_param(val, is_pct=False, is_atr=False):
    if val >= 100:
        return "NONE"
    elif is_pct:
        return f"{val*100:.0f}%"
    elif is_atr:
        return f"{val:.0f}x"
    return f"{val:.0f}"

# Top performers
print("\n" + "=" * 80)
print("TOP 20 BY TOTAL RETURN")
print("=" * 80)
top = results_df.nlargest(20, 'return')
print(f"\n{'Stop':>8} {'TP':>8} {'Days':>6} {'Trail':>8} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'DD':>7}")
print("-" * 90)
for _, r in top.iterrows():
    print(f"{format_param(r['sl_atr'], is_atr=True):>8} {format_param(r['tp_pct'], is_pct=True):>8} "
          f"{r['max_days']:>6.0f} {format_param(r['trail_pct'], is_pct=True):>8} | "
          f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} {r['max_dd']*100:>6.1f}%")

print("\n" + "=" * 80)
print("TOP 10 BY PROFIT FACTOR (with >50 trades)")
print("=" * 80)
filtered = results_df[(results_df['pf'] < 100) & (results_df['trades'] > 50)]
top_pf = filtered.nlargest(10, 'pf')
print(f"\n{'Stop':>8} {'TP':>8} {'Days':>6} {'Trail':>8} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'Trades':>7}")
print("-" * 90)
for _, r in top_pf.iterrows():
    print(f"{format_param(r['sl_atr'], is_atr=True):>8} {format_param(r['tp_pct'], is_pct=True):>8} "
          f"{r['max_days']:>6.0f} {format_param(r['trail_pct'], is_pct=True):>8} | "
          f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} {r['trades']:>7.0f}")

# Parameter sensitivity
print("\n" + "=" * 80)
print("PARAMETER SENSITIVITY")
print("=" * 80)

print("\n1. STOP-LOSS ATR:")
by_sl = results_df.groupby('sl_atr').agg({'return': 'mean', 'win_rate': 'mean', 'stop_rate': 'mean', 'pf': 'mean'}).round(4)
by_sl.index = [format_param(x, is_atr=True) for x in by_sl.index]
print(by_sl.to_string())

print("\n2. TAKE-PROFIT %:")
by_tp = results_df.groupby('tp_pct').agg({'return': 'mean', 'win_rate': 'mean', 'tp_rate': 'mean', 'pf': 'mean'}).round(4)
by_tp.index = [format_param(x, is_pct=True) for x in by_tp.index]
print(by_tp.to_string())

print("\n3. MAX HOLDING DAYS:")
by_hold = results_df.groupby('max_days').agg({'return': 'mean', 'win_rate': 'mean', 'time_rate': 'mean', 'pf': 'mean'}).round(4)
print(by_hold.to_string())

print("\n4. TRAILING STOP %:")
by_trail = results_df.groupby('trail_pct').agg({'return': 'mean', 'win_rate': 'mean', 'pf': 'mean'}).round(4)
by_trail.index = [format_param(x, is_pct=True) for x in by_trail.index]
print(by_trail.to_string())

# Best pure time-exit strategies (no stop, no take-profit)
print("\n" + "=" * 80)
print("BEST PURE TIME-EXIT STRATEGIES")
print("(No stop-loss, no take-profit - just hold for X days)")
print("=" * 80)
time_only = results_df[(results_df['sl_atr'] >= 100) & (results_df['tp_pct'] >= 100)]
if len(time_only) > 0:
    top_time = time_only.nlargest(10, 'return')
    print(f"\n{'Days':>6} {'Trail':>8} | {'Return':>9} {'Sharpe':>8} {'WR':>7} {'PF':>7} {'DD':>7} {'Trades':>7}")
    print("-" * 80)
    for _, r in top_time.iterrows():
        print(f"{r['max_days']:>6.0f} {format_param(r['trail_pct'], is_pct=True):>8} | "
              f"{r['return']*100:>+8.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% {r['pf']:>7.2f} "
              f"{r['max_dd']*100:>6.1f}% {r['trades']:>7.0f}")

# Best configuration
print("\n" + "=" * 80)
print("OPTIMAL CONFIGURATION")
print("=" * 80)

# Score configs
results_df['score'] = (
    results_df['return'] * 0.35 +
    results_df['sharpe'].clip(-2, 2) / 4 * 0.25 +
    (1 - results_df['max_dd']) * 0.25 +
    results_df['pf'].clip(0, 3) / 3 * 0.15
)

best = results_df.loc[results_df['score'].idxmax()]

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                   SCIENTIFICALLY OPTIMAL PARAMETERS                            │
├────────────────────────────────────────────────────────────────────────────────┤
│   Stop-Loss:           {format_param(best['sl_atr'], is_atr=True):>10}                                          │
│   Take-Profit:         {format_param(best['tp_pct'], is_pct=True):>10}                                          │
│   Max Holding Days:    {best['max_days']:>10.0f}                                          │
│   Trailing Stop:       {format_param(best['trail_pct'], is_pct=True):>10}                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│   PERFORMANCE:                                                                 │
│   Total Return:        {best['return']*100:>+10.1f}%                                         │
│   Sharpe Ratio:        {best['sharpe']:>+10.2f}                                          │
│   Win Rate:            {best['win_rate']*100:>10.1f}%                                         │
│   Profit Factor:       {best['pf']:>10.2f}                                          │
│   Max Drawdown:        {best['max_dd']*100:>10.1f}%                                         │
│   Total Trades:        {best['trades']:>10.0f}                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│   EXIT BREAKDOWN:                                                              │
│   Stop-Loss:           {best['stop_rate']*100:>10.1f}%                                         │
│   Take-Profit:         {best['tp_rate']*100:>10.1f}%                                         │
│   Time Exit:           {best['time_rate']*100:>10.1f}%                                         │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Key insight
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Best by category
best_return = results_df.loc[results_df['return'].idxmax()]
best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]

print(f"""
1. HIGHEST RETURN CONFIGURATION:
   - Stop: {format_param(best_return['sl_atr'], is_atr=True)}, TP: {format_param(best_return['tp_pct'], is_pct=True)}, 
     Days: {best_return['max_days']:.0f}, Trail: {format_param(best_return['trail_pct'], is_pct=True)}
   - Return: {best_return['return']*100:+.1f}%, Sharpe: {best_return['sharpe']:.2f}

2. HIGHEST SHARPE CONFIGURATION:
   - Stop: {format_param(best_sharpe['sl_atr'], is_atr=True)}, TP: {format_param(best_sharpe['tp_pct'], is_pct=True)}, 
     Days: {best_sharpe['max_days']:.0f}, Trail: {format_param(best_sharpe['trail_pct'], is_pct=True)}
   - Return: {best_sharpe['return']*100:+.1f}%, Sharpe: {best_sharpe['sharpe']:.2f}
""")

# Analysis of stop-loss impact
no_stop = results_df[results_df['sl_atr'] >= 100]['return'].mean()
with_stop = results_df[results_df['sl_atr'] < 100]['return'].mean()
print(f"""
3. STOP-LOSS IMPACT:
   - Average return WITHOUT stop-loss: {no_stop*100:+.2f}%
   - Average return WITH stop-loss:    {with_stop*100:+.2f}%
   - Difference: {(no_stop - with_stop)*100:+.2f}%
""")

no_tp = results_df[results_df['tp_pct'] >= 100]['return'].mean()
with_tp = results_df[results_df['tp_pct'] < 100]['return'].mean()
print(f"""4. TAKE-PROFIT IMPACT:
   - Average return WITHOUT take-profit: {no_tp*100:+.2f}%
   - Average return WITH take-profit:    {with_tp*100:+.2f}%
   - Difference: {(no_tp - with_tp)*100:+.2f}%
""")

# Save
os.makedirs('output/extended_optimization', exist_ok=True)
results_df.to_csv('output/extended_optimization/all_results.csv', index=False)
print(f"\nResults saved to output/extended_optimization/")

print("\n" + "=" * 80)
print("EXTENDED OPTIMIZATION COMPLETE")
print("=" * 80)
