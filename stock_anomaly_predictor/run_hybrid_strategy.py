#!/usr/bin/env python3
"""
HYBRID STRATEGY - Conditional Stop-Loss
========================================
Based on optimization findings:
- Trailing stop works well (+3.2% avg)
- Take-profit works excellently (+21% avg)
- Time exit is destroying returns (-11.6% avg)

SOLUTION: Add a time-conditional stop-loss
- No stop-loss early (let trade breathe)
- After X days, if losing, activate stop-loss
- This protects against holding losers forever
"""

import os
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 80)
print("HYBRID STRATEGY - Conditional Stop-Loss")
print("=" * 80)

# Test multiple hybrid configurations
CONFIGS = {
    'Conservative': {
        'stop_after_days': 30,       # Activate stop after 30 days
        'stop_if_down_pct': 0.05,    # Only if down > 5%
        'take_profit_pct': 0.20,
        'max_holding_days': 90,
        'trailing_stop_pct': 0.05,
    },
    'Moderate': {
        'stop_after_days': 45,
        'stop_if_down_pct': 0.10,    # Only if down > 10%
        'take_profit_pct': 0.20,
        'max_holding_days': 120,
        'trailing_stop_pct': 0.05,
    },
    'Aggressive': {
        'stop_after_days': 60,
        'stop_if_down_pct': 0.15,    # Only if down > 15%
        'take_profit_pct': 0.25,
        'max_holding_days': 180,
        'trailing_stop_pct': 0.07,
    },
    'TimeBasedSL': {
        'stop_after_days': 20,       # Early activation
        'stop_if_down_pct': 0.08,
        'take_profit_pct': 0.15,
        'max_holding_days': 60,
        'trailing_stop_pct': 0.04,
    }
}

STOCKS = {
    '1180.SR': 'Al Rajhi Bank', '1010.SR': 'Riyad Bank', '1050.SR': 'BSF',
    '1080.SR': 'ANB', '1140.SR': 'Albilad', '1150.SR': 'Alinma',
    '7010.SR': 'STC', '7020.SR': 'Mobily', '2010.SR': 'SABIC',
    '1211.SR': 'Maaden', '1320.SR': 'Steel Pipe', '3020.SR': 'Yamama',
    '4300.SR': 'Dar Al Arkan', '4310.SR': 'Emaar EC', '4190.SR': 'Jarir',
    '2280.SR': 'Almarai', '2050.SR': 'Savola', '8210.SR': 'Bupa',
    '2082.SR': 'ACWA Power', '2380.SR': 'Petro Rabigh',
}

# Fetch data
print("\n[1] FETCHING DATA...")
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

# Pre-calculate indicators and signals
indicators = {}
signals = {}
for ticker, df in stock_data.items():
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_pos = (close - (sma - 2*std)) / (4*std)
    
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean() if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    
    indicators[ticker] = {'rsi': rsi, 'bb_pos': bb_pos}
    signals[ticker] = ((rsi < 35) & (bb_pos < 0.3) & (vol_ratio > 1.5)).astype(int)

print("\n[2] TESTING HYBRID CONFIGURATIONS...")
print("=" * 80)

results = {}

for config_name, cfg in CONFIGS.items():
    cash = 1_000_000
    positions = {}
    trades = []
    equity = []
    
    for i, date in enumerate(trading_dates[:-1]):
        next_date = trading_dates[i + 1]
        
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
            current_return = price / pos['entry'] - 1
            
            # Update trailing stop
            if price > pos['high']:
                pos['high'] = price
                pos['trail'] = max(pos['trail'], price * (1 - cfg['trailing_stop_pct']))
            
            exit_reason = None
            
            # HYBRID: Time-conditional stop-loss
            if (holding >= cfg['stop_after_days'] and 
                current_return < -cfg['stop_if_down_pct']):
                exit_reason = 'conditional_stop'
            
            # Trailing stop (only if in profit)
            elif pos['trail'] > pos['entry'] and price <= pos['trail']:
                exit_reason = 'trailing_stop'
            
            # Take-profit
            elif price >= pos['tp']:
                exit_reason = 'take_profit'
            
            # Time exit
            elif holding >= cfg['max_holding_days']:
                exit_reason = 'time_exit'
            
            if exit_reason and ticker in next_opens:
                exit_price = next_opens[ticker] * 0.999
                pnl = (exit_price - pos['entry']) * pos['shares'] * 0.998
                
                trades.append({
                    'pnl': pnl,
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
            shares = int(cash * 0.05 / entry)
            if shares <= 0 or entry * shares > cash * 0.9:
                continue
            
            positions[ticker] = {
                'entry_date': date,
                'entry': entry,
                'shares': shares,
                'tp': entry * (1 + cfg['take_profit_pct']),
                'trail': entry,
                'high': entry
            }
            cash -= entry * shares * 1.001
        
        pos_val = sum(pos['shares'] * current_prices.get(t, pos['entry']) for t, pos in positions.items())
        equity.append(cash + pos_val)
    
    # Close remaining
    final = cash + sum(pos['shares'] * pos['entry'] for pos in positions.values())
    
    if len(trades) < 10:
        continue
    
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
    
    # Exit breakdown
    exit_stats = {}
    for reason in ['conditional_stop', 'trailing_stop', 'take_profit', 'time_exit']:
        reason_trades = [t for t in trades if t['reason'] == reason]
        if reason_trades:
            exit_stats[reason] = {
                'count': len(reason_trades),
                'pct': len(reason_trades) / len(trades) * 100,
                'avg_ret': np.mean([t['pnl_pct'] for t in reason_trades]) * 100
            }
    
    results[config_name] = {
        'config': cfg,
        'return': total_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'pf': pf,
        'max_dd': max_dd,
        'trades': len(trades),
        'exit_stats': exit_stats
    }
    
    print(f"\n{config_name}:")
    print(f"  Return: {total_ret*100:>+6.1f}%, Sharpe: {sharpe:>5.2f}, WR: {win_rate*100:>5.1f}%, PF: {pf:>5.2f}, DD: {max_dd*100:>5.1f}%")
    print(f"  Exit breakdown:")
    for reason, stats in exit_stats.items():
        print(f"    {reason:<18}: {stats['count']:>3} ({stats['pct']:>4.1f}%) avg: {stats['avg_ret']:>+5.1f}%")

# Summary comparison
print("\n" + "=" * 80)
print("HYBRID STRATEGY COMPARISON")
print("=" * 80)

print(f"\n{'Configuration':<18} {'Return':>10} {'Sharpe':>8} {'WR':>8} {'PF':>8} {'MaxDD':>8} {'Trades':>8}")
print("-" * 80)

for name, r in results.items():
    print(f"{name:<18} {r['return']*100:>+9.1f}% {r['sharpe']:>8.2f} {r['win_rate']*100:>7.1f}% {r['pf']:>8.2f} {r['max_dd']*100:>7.1f}% {r['trades']:>8}")

# Find best
best_name = max(results.keys(), key=lambda k: results[k]['return'])
best = results[best_name]

print("\n" + "=" * 80)
print(f"BEST CONFIGURATION: {best_name}")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZED HYBRID PARAMETERS                            │
├────────────────────────────────────────────────────────────────────────────────┤
│  Stop After Days:      {best['config']['stop_after_days']:>8}   (conditional activation)               │
│  Stop If Down %:       {best['config']['stop_if_down_pct']*100:>7.0f}%  (threshold for conditional SL)         │
│  Take-Profit:          {best['config']['take_profit_pct']*100:>7.0f}%                                          │
│  Max Holding Days:     {best['config']['max_holding_days']:>8}                                          │
│  Trailing Stop:        {best['config']['trailing_stop_pct']*100:>7.0f}%                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│  PERFORMANCE:                                                                  │
│  Total Return:         {best['return']*100:>+7.1f}%                                             │
│  Sharpe Ratio:         {best['sharpe']:>+7.2f}                                              │
│  Win Rate:             {best['win_rate']*100:>7.1f}%                                             │
│  Profit Factor:        {best['pf']:>7.2f}                                              │
│  Max Drawdown:         {best['max_dd']*100:>7.1f}%                                             │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Compare strategies
print("\n" + "=" * 80)
print("FULL STRATEGY EVOLUTION")
print("=" * 80)

print(f"""
┌────────────────────┬────────────┬────────────┬────────────┬────────────┐
│                    │  Original  │ Optimized  │   Hybrid   │  Previous  │
│     Metric         │(SL=2x,20d) │(NoSL,180d) │  (Best)    │ Best(30%TP)│
├────────────────────┼────────────┼────────────┼────────────┼────────────┤
│ Total Return       │    -6.9%   │    -1.1%   │  {best['return']*100:>+5.1f}%   │   +2.5%    │
│ Sharpe Ratio       │    -0.28   │    -0.00   │   {best['sharpe']:>+5.2f}   │   +0.15    │
│ Win Rate           │    41.0%   │    58.3%   │  {best['win_rate']*100:>5.1f}%   │   38.1%    │
│ Profit Factor      │     0.94   │     0.98   │   {best['pf']:>5.2f}   │    1.09    │
│ Max Drawdown       │    14.1%   │    10.3%   │   {best['max_dd']*100:>5.1f}%   │    7.3%    │
└────────────────────┴────────────┴────────────┴────────────┴────────────┘
""")

# Key insight about conditional stop
if 'conditional_stop' in best['exit_stats']:
    cs = best['exit_stats']['conditional_stop']
    print(f"""
KEY INSIGHT - Conditional Stop-Loss:
  - {cs['count']} trades ({cs['pct']:.1f}%) hit the conditional stop
  - Average return on conditional stops: {cs['avg_ret']:.1f}%
  - This prevents holding long-term losers!
""")

# Save results
os.makedirs('output/hybrid_strategy', exist_ok=True)
with open('output/hybrid_strategy/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to output/hybrid_strategy/")

print("\n" + "=" * 80)
print("HYBRID STRATEGY OPTIMIZATION COMPLETE")
print("=" * 80)
