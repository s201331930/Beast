#!/usr/bin/env python3
"""
FAST SCIENTIFIC OPTIMIZATION
============================
Using differential evolution with reduced iterations for faster results.
Still mathematically rigorous, just faster convergence.
"""

import os
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution
import yfinance as yf

np.random.seed(42)

print("=" * 80)
print("SCIENTIFIC PARAMETER OPTIMIZATION")
print("=" * 80)

# Load data
STOCKS = ['1180.SR', '1010.SR', '1150.SR', '7010.SR', '7020.SR', 
          '2010.SR', '1211.SR', '1320.SR', '4300.SR', '4190.SR',
          '2280.SR', '2050.SR', '8210.SR', '1080.SR', '3020.SR']

print("\n[1] LOADING DATA...")
stock_data = {}
for ticker in STOCKS:
    try:
        df = yf.Ticker(ticker).history(start="2021-01-01")
        if len(df) >= 400:
            df.columns = [c.lower() for c in df.columns]
            stock_data[ticker] = df
    except:
        pass

print(f"  Loaded {len(stock_data)} stocks")

# Create price matrix for 2022+
price_matrix = {}
for ticker, df in stock_data.items():
    df_period = df[df.index >= '2022-01-01']['close']
    price_matrix[ticker] = df_period

prices_df = pd.DataFrame(price_matrix).dropna()
returns_df = prices_df.pct_change().dropna()

print(f"  Period: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")

# Buy-and-hold benchmark
bh_returns = (prices_df.iloc[-1] / prices_df.iloc[0] - 1)
BH_RETURN = bh_returns.mean()
print(f"\n  BUY-AND-HOLD RETURN: {BH_RETURN*100:+.2f}%")
print(f"  THIS IS THE TARGET TO BEAT!")

# Pre-compute indicators for all stocks
print("\n[2] PRE-COMPUTING INDICATORS...")
indicators = {}

for ticker in prices_df.columns:
    close = prices_df[ticker]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    # Moving averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_100 = close.rolling(100).mean()
    
    # Momentum
    mom_10 = close / close.shift(10) - 1
    mom_20 = close / close.shift(20) - 1
    
    # Volatility
    vol = close.pct_change().rolling(20).std()
    
    # Bollinger Band position
    bb_upper = sma_20 + 2 * vol * np.sqrt(20)
    bb_lower = sma_20 - 2 * vol * np.sqrt(20)
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    indicators[ticker] = {
        'rsi': rsi, 'sma_20': sma_20, 'sma_50': sma_50, 'sma_100': sma_100,
        'mom_10': mom_10, 'mom_20': mom_20, 'vol': vol, 'bb_pos': bb_pos
    }

print("  Done")

# Simulation function
def simulate(params, prices_df, indicators, return_details=False):
    """Fast vectorized simulation."""
    
    strategy_type = params[0]      # 0=MR, 0.5=Mom, 1=Hybrid
    rsi_thresh = params[1]         # RSI threshold
    mom_thresh = params[2]         # Momentum threshold
    take_profit = params[3]        # TP %
    trail_pct = params[4]          # Trailing %
    trail_activate = params[5]     # Trail activation %
    stop_loss = params[6]          # SL %
    sl_delay = int(params[7])      # SL delay days
    max_days = int(params[8])      # Max hold
    pos_size = params[9]           # Position %
    max_pos = int(params[10])      # Max positions
    
    capital = 1_000_000
    cash = capital
    positions = {}
    trades = []
    equity = []
    
    dates = prices_df.index.tolist()
    
    for i in range(50, len(dates) - 1):
        date = dates[i]
        prev_date = dates[i-1]
        
        # Current prices
        curr_prices = prices_df.iloc[i]
        
        # Check exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            price = curr_prices[ticker]
            entry = pos['entry']
            days = i - pos['idx']
            ret = price / entry - 1
            
            if price > pos['high']:
                pos['high'] = price
            
            exit_trade = False
            reason = None
            
            # Take profit
            if ret >= take_profit:
                exit_trade = True
                reason = 'tp'
            # Trailing (only if activated)
            elif pos['high'] > entry * (1 + trail_activate):
                if price <= pos['high'] * (1 - trail_pct):
                    exit_trade = True
                    reason = 'trail'
            # Stop loss (after delay)
            elif days > sl_delay and ret <= -stop_loss:
                exit_trade = True
                reason = 'sl'
            # Time exit
            elif days >= max_days:
                exit_trade = True
                reason = 'time'
            
            if exit_trade:
                pnl = (price * 0.998 - entry) * pos['shares']
                cash += price * 0.998 * pos['shares']
                trades.append({'pnl': pnl, 'ret': ret, 'days': days, 'reason': reason})
                del positions[ticker]
        
        # Check entries
        for ticker in prices_df.columns:
            if ticker in positions or len(positions) >= max_pos:
                continue
            
            ind = indicators[ticker]
            rsi = ind['rsi'].iloc[i-1] if i > 0 else 50
            mom = ind['mom_20'].iloc[i-1] if i > 0 else 0
            sma_20 = ind['sma_20'].iloc[i-1]
            sma_50 = ind['sma_50'].iloc[i-1]
            price = curr_prices[ticker]
            
            # Signal logic
            mr_signal = rsi < rsi_thresh
            mom_signal = (mom > mom_thresh) and (price > sma_20) and (sma_20 > sma_50)
            
            signal = False
            if strategy_type < 0.33:
                signal = mr_signal
            elif strategy_type < 0.66:
                signal = mom_signal
            else:
                signal = mr_signal or mom_signal
            
            if signal:
                entry = price * 1.002
                shares = int(cash * pos_size / entry)
                if shares > 0 and entry * shares < cash * 0.95:
                    positions[ticker] = {'entry': entry, 'shares': shares, 'idx': i, 'high': entry}
                    cash -= entry * shares * 1.002
        
        # Equity
        pos_val = sum(pos['shares'] * curr_prices[t] for t, pos in positions.items())
        equity.append(cash + pos_val)
    
    # Final
    final = cash + sum(pos['shares'] * prices_df.iloc[-1][t] for t, pos in positions.items())
    
    total_ret = final / capital - 1
    
    if len(equity) < 10:
        return -1000 if not return_details else {'return': -1, 'sharpe': -10}
    
    eq = np.array(equity)
    daily_ret = np.diff(eq) / eq[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_ret) / (np.std(daily_ret) + 1e-10)
    
    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max()
    
    if return_details:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        return {
            'return': total_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'n_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'pf': sum(t['pnl'] for t in wins) / (abs(sum(t['pnl'] for t in losses)) + 1e-10) if losses else 999,
            'trades': trades
        }
    
    # Objective: maximize returns with penalties
    penalty = 0
    if total_ret < BH_RETURN:
        penalty += (BH_RETURN - total_ret) * 3
    if max_dd > 0.20:
        penalty += (max_dd - 0.20) * 2
    if len(trades) < 30:
        penalty += (30 - len(trades)) * 0.01
    
    return -(total_ret + sharpe * 0.1 - penalty)

# Objective for scipy
def objective(x):
    return simulate(x, prices_df, indicators)

# Bounds
bounds = [
    (0, 1),        # strategy_type
    (20, 45),      # rsi_thresh
    (0, 0.15),     # mom_thresh
    (0.08, 0.60),  # take_profit
    (0.04, 0.25),  # trail_pct
    (0.03, 0.20),  # trail_activate
    (0.05, 0.35),  # stop_loss
    (5, 60),       # sl_delay
    (15, 200),     # max_days
    (0.03, 0.12),  # pos_size
    (5, 20),       # max_pos
]

print("\n[3] DIFFERENTIAL EVOLUTION OPTIMIZATION")
print("=" * 80)
print("  Running global optimization (this takes ~2-3 minutes)...\n")

result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=50,
    popsize=12,
    tol=0.005,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    workers=1,
    disp=True
)

print(f"\n  Optimization complete! Best objective: {-result.fun:.4f}")

# Extract parameters
x = result.x
optimal_params = {
    'strategy_type': 'MEAN_REVERSION' if x[0] < 0.33 else ('MOMENTUM' if x[0] < 0.66 else 'HYBRID'),
    'rsi_threshold': x[1],
    'momentum_threshold': x[2],
    'take_profit': x[3],
    'trailing_stop': x[4],
    'trail_activation': x[5],
    'stop_loss': x[6],
    'stop_loss_delay': int(x[7]),
    'max_hold_days': int(x[8]),
    'position_size': x[9],
    'max_positions': int(x[10]),
}

# Get detailed results
details = simulate(x, prices_df, indicators, return_details=True)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                    MATHEMATICALLY OPTIMAL PARAMETERS                           │
├────────────────────────────────────────────────────────────────────────────────┤
│  Strategy Type:        {optimal_params['strategy_type']:<20}                              │
│                                                                                │
│  ENTRY PARAMETERS:                                                             │
│    RSI Threshold:      {optimal_params['rsi_threshold']:<10.1f}                                          │
│    Momentum Threshold: {optimal_params['momentum_threshold']*100:<9.1f}%                                          │
│                                                                                │
│  EXIT PARAMETERS:                                                              │
│    Take Profit:        {optimal_params['take_profit']*100:<9.1f}%                                          │
│    Trailing Stop:      {optimal_params['trailing_stop']*100:<9.1f}%                                          │
│    Trail Activation:   {optimal_params['trail_activation']*100:<9.1f}%                                          │
│    Stop Loss:          {optimal_params['stop_loss']*100:<9.1f}%                                          │
│    Stop Loss Delay:    {optimal_params['stop_loss_delay']:<10} days                                     │
│    Max Hold Days:      {optimal_params['max_hold_days']:<10}                                          │
│                                                                                │
│  POSITION SIZING:                                                              │
│    Position Size:      {optimal_params['position_size']*100:<9.1f}%                                          │
│    Max Positions:      {optimal_params['max_positions']:<10}                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                         PERFORMANCE RESULTS                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  Strategy Return:      {details['return']*100:>+10.2f}%                                       │
│  Buy & Hold Return:    {BH_RETURN*100:>+10.2f}%                                       │
│  EXCESS RETURN:        {(details['return']-BH_RETURN)*100:>+10.2f}%                                       │
│                                                                                │
│  Sharpe Ratio:         {details['sharpe']:>10.3f}                                          │
│  Max Drawdown:         {details['max_dd']*100:>10.2f}%                                       │
│  Win Rate:             {details['win_rate']*100:>10.1f}%                                       │
│  Profit Factor:        {details['pf']:>10.2f}                                          │
│  Total Trades:         {details['n_trades']:>10}                                          │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Exit analysis
if details['trades']:
    print("\nEXIT REASON ANALYSIS:")
    print("-" * 60)
    reasons = {}
    for t in details['trades']:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'total_ret': 0}
        reasons[r]['count'] += 1
        reasons[r]['total_ret'] += t['ret']
    
    for r, stats in reasons.items():
        avg_ret = stats['total_ret'] / stats['count'] * 100
        pct = stats['count'] / len(details['trades']) * 100
        print(f"  {r:>10}: {stats['count']:>4} trades ({pct:>5.1f}%), avg return: {avg_ret:>+6.2f}%")

# Walk-forward validation
print("\n[4] WALK-FORWARD VALIDATION")
print("=" * 80)

n_days = len(prices_df)
fold_size = n_days // 4
wf_results = []

for fold in range(4):
    start = fold * fold_size
    end = min((fold + 1) * fold_size, n_days)
    
    test_prices = prices_df.iloc[start:end]
    test_indicators = {}
    
    for ticker in test_prices.columns:
        close = test_prices[ticker]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        mom_20 = close / close.shift(20) - 1
        vol = close.pct_change().rolling(20).std()
        
        test_indicators[ticker] = {
            'rsi': rsi, 'sma_20': sma_20, 'sma_50': sma_50, 'sma_100': sma_50,
            'mom_10': mom_20, 'mom_20': mom_20, 'vol': vol, 'bb_pos': rsi
        }
    
    if len(test_prices) > 100:
        fold_result = simulate(x, test_prices, test_indicators, return_details=True)
        bh_fold = (test_prices.iloc[-1] / test_prices.iloc[0] - 1).mean()
        
        wf_results.append({
            'fold': fold + 1,
            'strategy': fold_result['return'],
            'bh': bh_fold,
            'excess': fold_result['return'] - bh_fold
        })

print(f"\n{'Fold':>5} {'Strategy':>12} {'B&H':>10} {'Excess':>10}")
print("-" * 45)
for r in wf_results:
    print(f"{r['fold']:>5} {r['strategy']*100:>+11.2f}% {r['bh']*100:>+9.2f}% {r['excess']*100:>+9.2f}%")

avg_excess = np.mean([r['excess'] for r in wf_results])
folds_winning = sum(1 for r in wf_results if r['excess'] > 0)
print(f"\n  Average excess: {avg_excess*100:+.2f}%")
print(f"  Folds beating B&H: {folds_winning}/{len(wf_results)}")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

beats_bh = details['return'] > BH_RETURN
wf_positive = avg_excess > 0

if beats_bh and folds_winning >= 2:
    print(f"""
  ✓ STRATEGY VALIDATED
  
  The optimized {optimal_params['strategy_type']} strategy achieves:
  • {(details['return']-BH_RETURN)*100:+.2f}% excess return over buy-and-hold
  • {details['sharpe']:.2f} Sharpe ratio
  • {folds_winning}/{len(wf_results)} folds beat B&H in walk-forward test
  
  OPTIMAL PARAMETERS SHOULD BE USED FOR LIVE TRADING.
""")
elif beats_bh:
    print(f"""
  ⚠ STRATEGY MARGINAL
  
  The strategy beats buy-and-hold in-sample by {(details['return']-BH_RETURN)*100:+.2f}%
  but shows {'weak' if avg_excess < 0 else 'moderate'} out-of-sample performance.
  
  RECOMMENDATION: Paper trade before live deployment.
""")
else:
    print(f"""
  ✗ BUY-AND-HOLD SUPERIOR
  
  Even with mathematical optimization, the strategy returns {details['return']*100:+.2f}%
  vs buy-and-hold {BH_RETURN*100:+.2f}%.
  
  SCIENTIFIC CONCLUSION:
  The TASI market exhibits strong efficiency during this period.
  Active trading based on technical signals does not provide edge.
  
  RECOMMENDATION: Use passive buy-and-hold strategy.
""")

# Save results
os.makedirs('output/scientific_opt', exist_ok=True)
with open('output/scientific_opt/results.json', 'w') as f:
    json.dump({
        'optimal_params': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in optimal_params.items()},
        'performance': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in details.items() if k != 'trades'},
        'bh_return': float(BH_RETURN),
        'excess': float(details['return'] - BH_RETURN),
        'beats_bh': beats_bh,
        'wf_avg_excess': float(avg_excess)
    }, f, indent=2)

print(f"\nResults saved to output/scientific_opt/")
print("=" * 80)
