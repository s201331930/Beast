#!/usr/bin/env python3
"""
MINIMAL FAST OPTIMIZATION
========================
Reduced complexity for quick convergence.
"""

import os, warnings, json
from datetime import datetime
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

print("=" * 80)
print("FAST PARAMETER OPTIMIZATION")
print("=" * 80)

# Minimal stock list
STOCKS = ['1180.SR', '1010.SR', '7010.SR', '2010.SR', '1211.SR', 
          '4300.SR', '2280.SR', '8210.SR', '1320.SR', '4190.SR']

print("\n[1] LOADING DATA...")
prices = {}
for t in STOCKS:
    try:
        df = yf.Ticker(t).history(start="2022-01-01")
        if len(df) > 200:
            prices[t] = df['Close']
    except:
        pass

prices_df = pd.DataFrame(prices).dropna()
print(f"  {len(prices_df)} days, {len(prices_df.columns)} stocks")

# Buy-and-hold
BH = (prices_df.iloc[-1] / prices_df.iloc[0] - 1).mean()
print(f"  Buy & Hold: {BH*100:+.2f}%")

# Pre-compute RSI
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l))

rsi_data = {t: calc_rsi(prices_df[t]) for t in prices_df.columns}
mom_data = {t: prices_df[t] / prices_df[t].shift(20) - 1 for t in prices_df.columns}

# Simulation with 5 key parameters
def simulate(params):
    tp, sl, max_days, rsi_th, pos_pct = params
    tp, sl = max(0.05, tp), max(0.05, sl)
    max_days = int(max(10, max_days))
    
    cash = 1_000_000
    positions = {}
    trades = []
    equity = []
    
    for i in range(30, len(prices_df) - 1):
        curr = prices_df.iloc[i]
        
        # Exits
        for t in list(positions.keys()):
            pos = positions[t]
            ret = curr[t] / pos['entry'] - 1
            days = i - pos['i']
            
            if pos['high'] < curr[t]:
                pos['high'] = curr[t]
            
            exit_trade = False
            if ret >= tp:
                exit_trade = True
            elif pos['high'] > pos['entry'] * 1.05 and curr[t] < pos['high'] * 0.93:
                exit_trade = True
            elif days > 20 and ret <= -sl:
                exit_trade = True
            elif days >= max_days:
                exit_trade = True
            
            if exit_trade:
                pnl = (curr[t] * 0.998 - pos['entry']) * pos['sh']
                cash += curr[t] * 0.998 * pos['sh']
                trades.append({'pnl': pnl, 'ret': ret})
                del positions[t]
        
        # Entries
        for t in prices_df.columns:
            if t in positions or len(positions) >= 10:
                continue
            
            rsi = rsi_data[t].iloc[i-1] if i > 0 else 50
            mom = mom_data[t].iloc[i-1] if i > 0 else 0
            
            # Entry signal: oversold OR strong momentum
            if rsi < rsi_th or (mom > 0.05 and rsi < 50):
                entry = curr[t] * 1.002
                shares = int(cash * pos_pct / entry)
                if shares > 0 and entry * shares < cash * 0.9:
                    positions[t] = {'entry': entry, 'sh': shares, 'i': i, 'high': entry}
                    cash -= entry * shares * 1.002
        
        pv = sum(pos['sh'] * curr[t] for t, pos in positions.items())
        equity.append(cash + pv)
    
    final = cash + sum(pos['sh'] * prices_df.iloc[-1][t] for t, pos in positions.items())
    ret = final / 1_000_000 - 1
    
    # Objective: maximize return, penalize if below B&H
    penalty = max(0, BH - ret) * 5
    return -(ret - penalty)

# Grid search for speed
print("\n[2] PARAMETER SEARCH...")
best = {'ret': -999}

# Test different strategies
configs = [
    # tp, sl, max_days, rsi_th, pos_pct
    (0.15, 0.10, 40, 35, 0.07),  # Balanced
    (0.20, 0.15, 60, 30, 0.08),  # Aggressive
    (0.25, 0.12, 80, 32, 0.06),  # Patient
    (0.30, 0.20, 100, 28, 0.05), # Very patient
    (0.35, 0.15, 120, 30, 0.05), # Max patience
    (0.40, 0.20, 150, 25, 0.04), # Ultra patient
    (0.50, 0.25, 180, 25, 0.04), # Max
    (0.20, 0.08, 50, 40, 0.10),  # Quick trades
    (0.15, 0.07, 30, 38, 0.08),  # Very quick
    (0.25, 0.10, 70, 35, 0.06),  # Medium
]

for cfg in configs:
    obj = simulate(cfg)
    ret = -obj + max(0, BH - (-obj - max(0, BH - (-obj)) * 5)) * 5  # Reverse penalty
    
    # Get actual return
    tp, sl, max_days, rsi_th, pos_pct = cfg
    
    # Recalculate
    cash = 1_000_000
    positions = {}
    trades = []
    
    for i in range(30, len(prices_df) - 1):
        curr = prices_df.iloc[i]
        
        for t in list(positions.keys()):
            pos = positions[t]
            ret_t = curr[t] / pos['entry'] - 1
            days = i - pos['i']
            
            if pos['high'] < curr[t]:
                pos['high'] = curr[t]
            
            exit_trade = False
            if ret_t >= tp:
                exit_trade = True
            elif pos['high'] > pos['entry'] * 1.05 and curr[t] < pos['high'] * 0.93:
                exit_trade = True
            elif days > 20 and ret_t <= -sl:
                exit_trade = True
            elif days >= max_days:
                exit_trade = True
            
            if exit_trade:
                pnl = (curr[t] * 0.998 - pos['entry']) * pos['sh']
                cash += curr[t] * 0.998 * pos['sh']
                trades.append({'pnl': pnl, 'ret': ret_t})
                del positions[t]
        
        for t in prices_df.columns:
            if t in positions or len(positions) >= 10:
                continue
            
            rsi = rsi_data[t].iloc[i-1] if i > 0 else 50
            mom = mom_data[t].iloc[i-1] if i > 0 else 0
            
            if rsi < rsi_th or (mom > 0.05 and rsi < 50):
                entry = curr[t] * 1.002
                shares = int(cash * pos_pct / entry)
                if shares > 0 and entry * shares < cash * 0.9:
                    positions[t] = {'entry': entry, 'sh': shares, 'i': i, 'high': entry}
                    cash -= entry * shares * 1.002
    
    final = cash + sum(pos['sh'] * prices_df.iloc[-1][t] for t, pos in positions.items())
    actual_ret = final / 1_000_000 - 1
    
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / len(trades) if trades else 0
    
    print(f"  TP={tp*100:>4.0f}% SL={sl*100:>4.0f}% Days={max_days:>3} RSI={rsi_th:>2} Pos={pos_pct*100:.0f}% -> Return: {actual_ret*100:>+6.2f}% WR: {wr*100:>5.1f}% Trades: {len(trades)}")
    
    if actual_ret > best['ret']:
        best = {'ret': actual_ret, 'cfg': cfg, 'trades': len(trades), 'wr': wr}

print(f"\n  BEST CONFIG: TP={best['cfg'][0]*100:.0f}%, SL={best['cfg'][1]*100:.0f}%, Days={best['cfg'][2]:.0f}, RSI={best['cfg'][3]:.0f}")

# Now use scipy to fine-tune around best
print("\n[3] FINE-TUNING WITH SCIPY...")

from scipy.optimize import minimize

x0 = list(best['cfg'])
bounds = [
    (best['cfg'][0] * 0.7, best['cfg'][0] * 1.3),
    (best['cfg'][1] * 0.7, best['cfg'][1] * 1.3),
    (best['cfg'][2] * 0.7, best['cfg'][2] * 1.3),
    (max(20, best['cfg'][3] - 10), min(45, best['cfg'][3] + 10)),
    (0.03, 0.12)
]

result = minimize(simulate, x0, method='L-BFGS-B', bounds=bounds, 
                  options={'maxiter': 50, 'disp': False})

opt = result.x
print(f"  Optimized: TP={opt[0]*100:.1f}%, SL={opt[1]*100:.1f}%, Days={int(opt[2])}, RSI={opt[3]:.1f}")

# Final simulation with optimal params
tp, sl, max_days, rsi_th, pos_pct = opt
max_days = int(max(10, max_days))

cash = 1_000_000
positions = {}
trades = []
equity = []

for i in range(30, len(prices_df) - 1):
    curr = prices_df.iloc[i]
    
    for t in list(positions.keys()):
        pos = positions[t]
        ret = curr[t] / pos['entry'] - 1
        days = i - pos['i']
        
        if pos['high'] < curr[t]:
            pos['high'] = curr[t]
        
        exit_trade = False
        reason = ''
        if ret >= tp:
            exit_trade, reason = True, 'tp'
        elif pos['high'] > pos['entry'] * 1.05 and curr[t] < pos['high'] * 0.93:
            exit_trade, reason = True, 'trail'
        elif days > 20 and ret <= -sl:
            exit_trade, reason = True, 'sl'
        elif days >= max_days:
            exit_trade, reason = True, 'time'
        
        if exit_trade:
            pnl = (curr[t] * 0.998 - pos['entry']) * pos['sh']
            cash += curr[t] * 0.998 * pos['sh']
            trades.append({'pnl': pnl, 'ret': ret, 'reason': reason})
            del positions[t]
    
    for t in prices_df.columns:
        if t in positions or len(positions) >= 10:
            continue
        
        rsi = rsi_data[t].iloc[i-1] if i > 0 else 50
        mom = mom_data[t].iloc[i-1] if i > 0 else 0
        
        if rsi < rsi_th or (mom > 0.05 and rsi < 50):
            entry = curr[t] * 1.002
            shares = int(cash * pos_pct / entry)
            if shares > 0 and entry * shares < cash * 0.9:
                positions[t] = {'entry': entry, 'sh': shares, 'i': i, 'high': entry}
                cash -= entry * shares * 1.002
    
    pv = sum(pos['sh'] * curr[t] for t, pos in positions.items())
    equity.append(cash + pv)

final = cash + sum(pos['sh'] * prices_df.iloc[-1][t] for t, pos in positions.items())
final_ret = final / 1_000_000 - 1

eq = np.array(equity)
sharpe = np.sqrt(252) * np.mean(np.diff(eq) / eq[:-1]) / (np.std(np.diff(eq) / eq[:-1]) + 1e-10)
peak = np.maximum.accumulate(eq)
max_dd = ((peak - eq) / peak).max()

wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] <= 0]
wr = len(wins) / len(trades) if trades else 0
pf = sum(t['pnl'] for t in wins) / (abs(sum(t['pnl'] for t in losses)) + 1e-10) if losses else 999

print(f"""
================================================================================
FINAL RESULTS
================================================================================

OPTIMAL PARAMETERS:
  Take Profit:     {tp*100:.1f}%
  Stop Loss:       {sl*100:.1f}%
  Max Hold Days:   {max_days}
  RSI Threshold:   {rsi_th:.1f}
  Position Size:   {pos_pct*100:.1f}%

PERFORMANCE:
  Strategy Return: {final_ret*100:+.2f}%
  Buy & Hold:      {BH*100:+.2f}%
  EXCESS:          {(final_ret - BH)*100:+.2f}%
  
  Sharpe Ratio:    {sharpe:.3f}
  Max Drawdown:    {max_dd*100:.2f}%
  Win Rate:        {wr*100:.1f}%
  Profit Factor:   {pf:.2f}
  Total Trades:    {len(trades)}

EXIT BREAKDOWN:
""")

reasons = {}
for t in trades:
    r = t.get('reason', 'other')
    if r not in reasons:
        reasons[r] = {'n': 0, 'ret': 0}
    reasons[r]['n'] += 1
    reasons[r]['ret'] += t['ret']

for r, s in reasons.items():
    print(f"  {r:>8}: {s['n']:>4} trades, avg return: {s['ret']/s['n']*100:>+6.2f}%")

print("\n" + "=" * 80)
if final_ret > BH:
    print(f"SUCCESS: Strategy beats buy-and-hold by {(final_ret - BH)*100:+.2f}%")
else:
    print(f"RESULT: Strategy returns {final_ret*100:+.2f}% vs B&H {BH*100:+.2f}%")
print("=" * 80)

os.makedirs('output/minimal_opt', exist_ok=True)
with open('output/minimal_opt/results.json', 'w') as f:
    json.dump({
        'params': {'tp': tp, 'sl': sl, 'max_days': max_days, 'rsi': rsi_th, 'pos': pos_pct},
        'return': final_ret, 'bh': BH, 'excess': final_ret - BH,
        'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': wr, 'pf': pf, 'trades': len(trades)
    }, f, indent=2)
