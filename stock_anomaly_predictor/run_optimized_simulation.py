#!/usr/bin/env python3
"""
OPTIMIZED PORTFOLIO SIMULATION
==============================
Using scientifically optimized parameters from our grid search.

OPTIMIZED PARAMETERS:
- Stop-Loss: 3.0x ATR (was 2.0x)
- Take-Profit: 30% (was 10%)  
- Max Holding Days: 60 (was 20)
- Trailing Stop: 5% (was 3%)
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 80)
print("OPTIMIZED PORTFOLIO SIMULATION - TASI")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# OPTIMIZED PARAMETERS
CONFIG = {
    'initial_capital': 1_000_000,
    'max_position_pct': 0.05,
    'max_positions': 15,
    
    # OPTIMIZED VALUES
    'stop_loss_atr': 3.0,      # Was 2.0
    'take_profit_pct': 0.30,   # Was 0.10 - THIS IS KEY!
    'max_holding_days': 60,    # Was 20
    'trailing_stop_pct': 0.05, # Was 0.03
    
    'commission_pct': 0.001,
    'slippage_pct': 0.001,
}

print(f"""
OPTIMIZED PARAMETERS:
─────────────────────────────────────────────────────────────────────────────────
  Stop-Loss ATR:       {CONFIG['stop_loss_atr']}x   (was 2.0x)
  Take-Profit:         {CONFIG['take_profit_pct']*100:.0f}%  (was 10%)  ← KEY CHANGE!
  Max Holding Days:    {CONFIG['max_holding_days']}   (was 20)
  Trailing Stop:       {CONFIG['trailing_stop_pct']*100:.0f}%   (was 3%)
─────────────────────────────────────────────────────────────────────────────────
""")

# Full TASI universe
STOCKS = {
    '1180.SR': 'Al Rajhi Bank',
    '7010.SR': 'STC',
    '7020.SR': 'Mobily',
    '1150.SR': 'Alinma Bank',
    '4300.SR': 'Dar Al Arkan',
    '1320.SR': 'Saudi Steel Pipe',
    '1211.SR': 'Maaden',
    '3020.SR': 'Yamama Cement',
    '1010.SR': 'Riyad Bank',
    '4190.SR': 'Jarir Marketing',
    '1050.SR': 'Banque Saudi Fransi',
    '1060.SR': 'Saudi Awwal Bank',
    '1080.SR': 'Arab National Bank',
    '1140.SR': 'Bank Albilad',
    '2010.SR': 'SABIC',
    '2222.SR': 'Saudi Aramco',
    '4310.SR': 'Emaar EC',
    '1304.SR': 'Yamamah Steel',
    '1321.SR': 'East Pipes',
    '1322.SR': 'Almasane Mining',
    '8210.SR': 'Bupa Arabia',
    '2280.SR': 'Almarai',
    '2050.SR': 'Savola',
    '4009.SR': 'ME Healthcare',
    '2380.SR': 'Petro Rabigh',
}

print(f"[1] FETCHING DATA ({len(STOCKS)} stocks)...")
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

# Get trading dates
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index.tolist())
trading_dates = sorted(all_dates)
print(f"  Trading days: {len(trading_dates)}")
print(f"  Period: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")

# Calculate indicators
print("\n[2] CALCULATING INDICATORS...")
indicators = {}
signals = {}

for ticker, df in stock_data.items():
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
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
    
    indicators[ticker] = {'rsi': rsi, 'atr': atr, 'bb_pos': bb_pos, 'vol_ratio': vol_ratio}
    
    # Signal: RSI < 35, near lower BB, good volume
    signals[ticker] = ((rsi < 35) & (bb_pos < 0.3) & (vol_ratio > 1.5)).astype(int)

print("  Done")

# Run simulation
print("\n[3] RUNNING OPTIMIZED SIMULATION...")
cash = CONFIG['initial_capital']
positions = {}
trades = []
equity_history = []

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
            pos['trail'] = max(pos['trail'], price * (1 - CONFIG['trailing_stop_pct']))
        
        stop = max(pos['stop'], pos['trail'])
        
        exit_reason = None
        if price <= stop:
            exit_reason = 'stop_loss'
        elif price >= pos['tp']:
            exit_reason = 'take_profit'
        elif holding >= CONFIG['max_holding_days']:
            exit_reason = 'time_exit'
        
        if exit_reason and ticker in next_opens:
            exit_price = next_opens[ticker] * (1 - CONFIG['slippage_pct'])
            gross = (exit_price - pos['entry']) * pos['shares']
            commission = exit_price * pos['shares'] * CONFIG['commission_pct']
            net_pnl = gross - commission
            
            trades.append({
                'ticker': ticker,
                'name': STOCKS.get(ticker, ticker),
                'entry_date': pos['entry_date'],
                'exit_date': next_date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': net_pnl,
                'pnl_pct': exit_price / pos['entry'] - 1,
                'holding_days': holding,
                'exit_reason': exit_reason
            })
            
            cash += exit_price * pos['shares'] - commission
            del positions[ticker]
    
    # Check entries
    for ticker in stock_data.keys():
        if ticker in positions or len(positions) >= CONFIG['max_positions']:
            continue
        if date not in signals[ticker].index or signals[ticker].loc[date] != 1:
            continue
        if ticker not in next_opens:
            continue
        
        entry = next_opens[ticker] * (1 + CONFIG['slippage_pct'])
        atr_val = indicators[ticker]['atr'].loc[date] if date in indicators[ticker]['atr'].index else entry * 0.02
        if pd.isna(atr_val):
            atr_val = entry * 0.02
        
        shares = int(min(cash * CONFIG['max_position_pct'], cash * 0.9) / entry)
        if shares <= 0:
            continue
        
        cost = entry * shares * (1 + CONFIG['commission_pct'])
        if cost > cash:
            continue
        
        positions[ticker] = {
            'entry_date': next_date,
            'entry': entry,
            'shares': shares,
            'stop': entry - atr_val * CONFIG['stop_loss_atr'],
            'tp': entry * (1 + CONFIG['take_profit_pct']),
            'trail': entry - atr_val * CONFIG['stop_loss_atr'],
            'high': entry
        }
        cash -= cost
    
    # Record equity
    pos_val = sum(pos['shares'] * current_prices.get(t, pos['entry']) for t, pos in positions.items())
    equity_history.append({
        'date': date,
        'cash': cash,
        'positions': pos_val,
        'total': cash + pos_val,
        'num_positions': len(positions)
    })
    
    # Progress
    if (i + 1) % 100 == 0 or i == 0:
        total = cash + pos_val
        pnl = total - CONFIG['initial_capital']
        print(f"  {date.strftime('%Y-%m-%d')}: {total:,.0f} SAR ({pnl:+,.0f}, {pnl/CONFIG['initial_capital']*100:+.1f}%)")

# Close remaining positions
print("\n[4] CLOSING REMAINING POSITIONS...")
final_date = trading_dates[-1]
for ticker, pos in list(positions.items()):
    if ticker in stock_data and final_date in stock_data[ticker].index:
        exit_price = stock_data[ticker].loc[final_date, 'close']
    else:
        exit_price = pos['entry']
    
    pnl = (exit_price - pos['entry']) * pos['shares']
    trades.append({
        'ticker': ticker,
        'name': STOCKS.get(ticker, ticker),
        'entry_date': pos['entry_date'],
        'exit_date': final_date,
        'entry_price': pos['entry'],
        'exit_price': exit_price,
        'shares': pos['shares'],
        'pnl': pnl - exit_price * pos['shares'] * CONFIG['commission_pct'],
        'pnl_pct': exit_price / pos['entry'] - 1,
        'holding_days': (final_date - pos['entry_date']).days,
        'exit_reason': 'simulation_end'
    })
    cash += exit_price * pos['shares'] * (1 - CONFIG['commission_pct'])

# Calculate results
print("\n" + "=" * 80)
print("OPTIMIZED SIMULATION RESULTS")
print("=" * 80)

initial = CONFIG['initial_capital']
final = cash
total_return = final / initial - 1

equity_df = pd.DataFrame(equity_history)

# Trade stats
trades_df = pd.DataFrame(trades)
winning = trades_df[trades_df['pnl'] > 0]
losing = trades_df[trades_df['pnl'] <= 0]

win_rate = len(winning) / len(trades_df) if len(trades_df) > 0 else 0
profit_factor = winning['pnl'].sum() / abs(losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')

# Risk metrics
if len(equity_df) > 1:
    equity_df['return'] = equity_df['total'].pct_change()
    daily_rets = equity_df['return'].dropna()
    sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std() if daily_rets.std() > 0 else 0
    
    peak = equity_df['total'].cummax()
    dd = (peak - equity_df['total']) / peak
    max_dd = dd.max()
else:
    sharpe = 0
    max_dd = 0

years = len(equity_df) / 252
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

# Exit breakdown
stop_loss_count = len(trades_df[trades_df['exit_reason'] == 'stop_loss'])
take_profit_count = len(trades_df[trades_df['exit_reason'] == 'take_profit'])
time_exit_count = len(trades_df[trades_df['exit_reason'] == 'time_exit'])

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZED PORTFOLIO PERFORMANCE                           │
├────────────────────────────────────────────────────────────────────────────────┤
│  Initial Capital:      {initial:>15,.0f} SAR                                  │
│  Final Value:          {final:>15,.0f} SAR                                  │
│  Total P&L:            {final-initial:>+15,.0f} SAR                                  │
│  Total Return:         {total_return*100:>+14.1f}%                                     │
│  Annual Return:        {annual_return*100:>+14.1f}%                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│  Sharpe Ratio:         {sharpe:>15.2f}                                          │
│  Max Drawdown:         {max_dd*100:>14.1f}%                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│  Total Trades:         {len(trades_df):>15}                                          │
│  Win Rate:             {win_rate*100:>14.1f}%                                     │
│  Profit Factor:        {profit_factor:>15.2f}                                          │
│  Avg Win:              {winning['pnl'].mean() if len(winning) > 0 else 0:>+15,.0f} SAR                                  │
│  Avg Loss:             {losing['pnl'].mean() if len(losing) > 0 else 0:>+15,.0f} SAR                                  │
│  Avg Holding Days:     {trades_df['holding_days'].mean():>15.1f}                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│  EXIT BREAKDOWN:                                                               │
│  Stop-Loss:            {stop_loss_count:>5} ({stop_loss_count/len(trades_df)*100 if len(trades_df) > 0 else 0:>5.1f}%)                                             │
│  Take-Profit:          {take_profit_count:>5} ({take_profit_count/len(trades_df)*100 if len(trades_df) > 0 else 0:>5.1f}%)                                             │
│  Time Exit:            {time_exit_count:>5} ({time_exit_count/len(trades_df)*100 if len(trades_df) > 0 else 0:>5.1f}%)                                             │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON: ORIGINAL vs OPTIMIZED")
print("=" * 80)

original = {
    'return': -0.069,
    'sharpe': -0.28,
    'max_dd': 0.141,
    'win_rate': 0.41,
    'pf': 0.94,
    'trades': 1150,
    'stop_loss_rate': 0.714
}

print(f"""
┌───────────────────────┬────────────────┬────────────────┬────────────────┐
│       Metric          │    Original    │   Optimized    │   Improvement  │
├───────────────────────┼────────────────┼────────────────┼────────────────┤
│ Total Return          │   {original['return']*100:>+8.1f}%     │   {total_return*100:>+8.1f}%     │   {(total_return-original['return'])*100:>+8.1f}%     │
│ Sharpe Ratio          │   {original['sharpe']:>+8.2f}       │   {sharpe:>+8.2f}       │   {sharpe-original['sharpe']:>+8.2f}       │
│ Max Drawdown          │    {original['max_dd']*100:>8.1f}%     │    {max_dd*100:>8.1f}%     │   {(original['max_dd']-max_dd)*100:>+8.1f}%     │
│ Win Rate              │    {original['win_rate']*100:>8.1f}%     │    {win_rate*100:>8.1f}%     │   {(win_rate-original['win_rate'])*100:>+8.1f}%     │
│ Profit Factor         │    {original['pf']:>8.2f}       │    {profit_factor:>8.2f}       │   {profit_factor-original['pf']:>+8.2f}       │
│ Total Trades          │    {original['trades']:>8}       │    {len(trades_df):>8}       │   {len(trades_df)-original['trades']:>+8}       │
│ Stop-Loss Rate        │    {original['stop_loss_rate']*100:>8.1f}%     │    {stop_loss_count/len(trades_df)*100 if len(trades_df) > 0 else 0:>8.1f}%     │   {(stop_loss_count/len(trades_df) if len(trades_df) > 0 else 0 - original['stop_loss_rate'])*100:>+8.1f}%     │
└───────────────────────┴────────────────┴────────────────┴────────────────┘
""")

# Top trades
print("\n" + "-" * 80)
print("TOP 10 WINNING TRADES")
print("-" * 80)
top_wins = trades_df.nlargest(10, 'pnl')
for _, t in top_wins.iterrows():
    print(f"  {t['ticker']:<10} {t['name'][:20]:<20} {t['entry_date'].strftime('%Y-%m-%d')} → {t['exit_date'].strftime('%Y-%m-%d')} "
          f"{t['pnl_pct']*100:>+6.1f}% {t['pnl']:>+10,.0f} SAR ({t['exit_reason']})")

# Exit reason analysis
print("\n" + "-" * 80)
print("EXIT REASON P&L ANALYSIS")
print("-" * 80)
by_exit = trades_df.groupby('exit_reason').agg({
    'ticker': 'count',
    'pnl': ['sum', 'mean'],
    'pnl_pct': 'mean'
})
by_exit.columns = ['Count', 'Total PnL', 'Avg PnL', 'Avg %']
print(by_exit.to_string())

# Save results
os.makedirs('output/optimized_simulation', exist_ok=True)
equity_df.to_csv('output/optimized_simulation/equity_curve.csv', index=False)
trades_df.to_csv('output/optimized_simulation/trades.csv', index=False)

results = {
    'parameters': CONFIG,
    'total_return': float(total_return),
    'annual_return': float(annual_return),
    'sharpe_ratio': float(sharpe),
    'max_drawdown': float(max_dd),
    'win_rate': float(win_rate),
    'profit_factor': float(profit_factor),
    'total_trades': len(trades_df),
    'stop_loss_rate': stop_loss_count / len(trades_df) if len(trades_df) > 0 else 0,
    'take_profit_rate': take_profit_count / len(trades_df) if len(trades_df) > 0 else 0,
    'time_exit_rate': time_exit_count / len(trades_df) if len(trades_df) > 0 else 0
}

with open('output/optimized_simulation/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to output/optimized_simulation/")

# Final verdict
print("\n" + "=" * 80)
print("STRATEGY VERDICT")
print("=" * 80)

checks = []
if total_return > 0:
    checks.append(("✓", f"Positive return: {total_return*100:+.1f}%"))
else:
    checks.append(("✗", f"Negative return: {total_return*100:+.1f}%"))

if sharpe > 0.5:
    checks.append(("✓", f"Good Sharpe ratio: {sharpe:.2f}"))
elif sharpe > 0:
    checks.append(("⚠️", f"Marginal Sharpe: {sharpe:.2f}"))
else:
    checks.append(("✗", f"Negative Sharpe: {sharpe:.2f}"))

if max_dd < 0.15:
    checks.append(("✓", f"Low drawdown: {max_dd*100:.1f}%"))
else:
    checks.append(("⚠️", f"High drawdown: {max_dd*100:.1f}%"))

if win_rate > 0.40:
    checks.append(("✓", f"Acceptable win rate: {win_rate*100:.1f}%"))
else:
    checks.append(("⚠️", f"Low win rate: {win_rate*100:.1f}%"))

if profit_factor > 1.0:
    checks.append(("✓", f"Profitable: PF = {profit_factor:.2f}"))
else:
    checks.append(("✗", f"Unprofitable: PF = {profit_factor:.2f}"))

for sym, msg in checks:
    print(f"  {sym} {msg}")

passed = sum(1 for s, _ in checks if s == "✓")
print(f"\n  Score: {passed}/{len(checks)} checks passed")

if passed >= 4:
    print("\n  🏆 STRATEGY VALIDATED - Ready for paper trading!")
elif passed >= 3:
    print("\n  ⚠️  STRATEGY IMPROVED - Close to validation")
else:
    print("\n  ❌ STRATEGY NEEDS MORE WORK")

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
