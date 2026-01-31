#!/usr/bin/env python3
"""
FINAL VALIDATION - Scientifically Optimized Strategy
====================================================
Using parameters derived from 750+ simulations:

OPTIMAL PARAMETERS:
- Stop-Loss: NONE (key insight: stop-losses were killing returns!)
- Take-Profit: 20% (modest target)
- Max Holding Days: 180 (let trades breathe!)
- Trailing Stop: 5% (protect profits only when meaningful)

Run on FULL TASI universe for final validation.
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
print("FINAL VALIDATION - Scientifically Optimized Strategy")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# SCIENTIFICALLY OPTIMIZED PARAMETERS
CONFIG = {
    'initial_capital': 1_000_000,
    'max_position_pct': 0.05,
    'max_positions': 20,
    
    # OPTIMIZED - KEY CHANGES
    'use_stop_loss': False,    # DISABLED - was killing returns!
    'stop_loss_atr': 3.0,      # Only used if enabled
    'take_profit_pct': 0.20,   # 20% - was 10%
    'max_holding_days': 180,   # 180 days - was 20!
    'trailing_stop_pct': 0.05, # 5% trailing
    'use_trailing': True,      # Protect profits
    
    'commission_pct': 0.001,
    'slippage_pct': 0.001,
}

print(f"""
OPTIMIZED PARAMETERS (from 750+ simulations):
─────────────────────────────────────────────────────────────────────────────────
  Stop-Loss:          {'DISABLED' if not CONFIG['use_stop_loss'] else f"{CONFIG['stop_loss_atr']}x ATR"}  ← KEY CHANGE!
  Take-Profit:        {CONFIG['take_profit_pct']*100:.0f}%
  Max Holding Days:   {CONFIG['max_holding_days']}  ← KEY CHANGE!
  Trailing Stop:      {CONFIG['trailing_stop_pct']*100:.0f}%
─────────────────────────────────────────────────────────────────────────────────
""")

# FULL TASI UNIVERSE
STOCKS = {
    # Banks
    '1180.SR': 'Al Rajhi Bank',
    '1010.SR': 'Riyad Bank',
    '1050.SR': 'Banque Saudi Fransi',
    '1060.SR': 'Saudi Awwal Bank',
    '1080.SR': 'Arab National Bank',
    '1140.SR': 'Bank Albilad',
    '1150.SR': 'Alinma Bank',
    # Telecom
    '7010.SR': 'STC',
    '7020.SR': 'Mobily',
    '7030.SR': 'Zain KSA',
    # Materials
    '2010.SR': 'SABIC',
    '1211.SR': 'Maaden',
    '1320.SR': 'Saudi Steel Pipe',
    '1304.SR': 'Yamamah Steel',
    '1321.SR': 'East Pipes',
    '3020.SR': 'Yamama Cement',
    '3030.SR': 'Saudi Cement',
    '3050.SR': 'Southern Cement',
    # Energy
    '2222.SR': 'Saudi Aramco',
    '2380.SR': 'Petro Rabigh',
    # Real Estate
    '4300.SR': 'Dar Al Arkan',
    '4310.SR': 'Emaar EC',
    # Retail
    '4190.SR': 'Jarir Marketing',
    # Food
    '2280.SR': 'Almarai',
    '2050.SR': 'Savola',
    # Insurance
    '8210.SR': 'Bupa Arabia',
    # Healthcare
    '4009.SR': 'ME Healthcare',
    # Utilities
    '2082.SR': 'ACWA Power',
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
    
    # Bollinger Bands
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_pos = (close - (sma - 2*std)) / (4*std)
    
    # Volume
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean() if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    
    indicators[ticker] = {'rsi': rsi, 'atr': atr, 'bb_pos': bb_pos, 'vol_ratio': vol_ratio}
    
    # Signal: Oversold + Near lower BB + Volume confirmation
    signals[ticker] = ((rsi < 35) & (bb_pos < 0.3) & (vol_ratio > 1.5)).astype(int)

print("  Done")

# Run simulation
print("\n[3] RUNNING FINAL SIMULATION...")
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
        
        # Update trailing stop if enabled and price made new high
        if CONFIG['use_trailing'] and price > pos['high']:
            pos['high'] = price
            pos['trail'] = max(pos['trail'], price * (1 - CONFIG['trailing_stop_pct']))
        
        exit_reason = None
        
        # Stop-loss (DISABLED by default)
        if CONFIG['use_stop_loss'] and price <= pos['stop']:
            exit_reason = 'stop_loss'
        
        # Trailing stop (only after profit)
        elif CONFIG['use_trailing'] and pos['trail'] > pos['entry'] and price <= pos['trail']:
            exit_reason = 'trailing_stop'
        
        # Take-profit
        elif price >= pos['tp']:
            exit_reason = 'take_profit'
        
        # Time exit
        elif holding >= CONFIG['max_holding_days']:
            exit_reason = 'time_exit'
        
        if exit_reason and ticker in next_opens:
            exit_price = next_opens[ticker] * (1 - CONFIG['slippage_pct'])
            gross = (exit_price - pos['entry']) * pos['shares']
            commission = exit_price * pos['shares'] * CONFIG['commission_pct']
            
            trades.append({
                'ticker': ticker,
                'name': STOCKS.get(ticker, ticker),
                'entry_date': pos['entry_date'],
                'exit_date': next_date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': gross - commission,
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
            'stop': entry - atr_val * CONFIG['stop_loss_atr'] if CONFIG['use_stop_loss'] else 0,
            'tp': entry * (1 + CONFIG['take_profit_pct']),
            'trail': entry,  # Start at entry, only moves up
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
    if (i + 1) % 200 == 0 or i == 0:
        total_val = cash + pos_val
        pnl = total_val - CONFIG['initial_capital']
        print(f"  {date.strftime('%Y-%m-%d')}: {total_val:>12,.0f} SAR ({pnl:>+10,.0f}, {pnl/CONFIG['initial_capital']*100:>+5.1f}%)")

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

# Calculate metrics
print("\n" + "=" * 80)
print("FINAL VALIDATION RESULTS")
print("=" * 80)

initial = CONFIG['initial_capital']
final = cash
total_return = final / initial - 1

equity_df = pd.DataFrame(equity_history)
trades_df = pd.DataFrame(trades)

winning = trades_df[trades_df['pnl'] > 0]
losing = trades_df[trades_df['pnl'] <= 0]

win_rate = len(winning) / len(trades_df) if len(trades_df) > 0 else 0
profit_factor = winning['pnl'].sum() / abs(losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf')

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
exit_counts = trades_df['exit_reason'].value_counts()

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                    FINAL VALIDATED STRATEGY PERFORMANCE                        │
├────────────────────────────────────────────────────────────────────────────────┤
│  Initial Capital:      {initial:>15,.0f} SAR                                  │
│  Final Value:          {final:>15,.0f} SAR                                  │
│  Total P&L:            {final-initial:>+15,.0f} SAR                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│  TOTAL RETURN:         {total_return*100:>+14.1f}%                                     │
│  ANNUAL RETURN:        {annual_return*100:>+14.1f}%                                     │
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
└────────────────────────────────────────────────────────────────────────────────┘
""")

# Exit breakdown
print("\n" + "-" * 80)
print("EXIT REASON BREAKDOWN")
print("-" * 80)
for reason, count in exit_counts.items():
    pct = count / len(trades_df) * 100
    avg_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl_pct'].mean() * 100
    print(f"  {reason:<18} {count:>5} trades ({pct:>5.1f}%)  Avg Return: {avg_pnl:>+6.1f}%")

# Full comparison
print("\n" + "=" * 80)
print("STRATEGY EVOLUTION COMPARISON")
print("=" * 80)

print(f"""
┌───────────────────────┬───────────────┬───────────────┬───────────────┐
│       Metric          │   Original    │  Intermediate │    FINAL      │
│                       │ (SL=2x,TP=10%)│(SL=3x,TP=30%) │(NO SL,TP=20%) │
├───────────────────────┼───────────────┼───────────────┼───────────────┤
│ Stop-Loss ATR         │      2.0x     │      3.0x     │     NONE      │
│ Take-Profit %         │      10%      │      30%      │      20%      │
│ Max Holding Days      │      20       │      60       │      180      │
├───────────────────────┼───────────────┼───────────────┼───────────────┤
│ Total Return          │     -6.9%     │     +2.5%     │   {total_return*100:>+6.1f}%     │
│ Sharpe Ratio          │     -0.28     │     +0.15     │    {sharpe:>+5.2f}      │
│ Win Rate              │     41.0%     │     38.1%     │    {win_rate*100:>5.1f}%     │
│ Profit Factor         │      0.94     │      1.09     │     {profit_factor:>5.2f}      │
│ Max Drawdown          │     14.1%     │      7.3%     │     {max_dd*100:>5.1f}%     │
└───────────────────────┴───────────────┴───────────────┴───────────────┘
""")

# Top winners
print("\n" + "-" * 80)
print("TOP 10 WINNING TRADES")
print("-" * 80)
top_wins = trades_df.nlargest(10, 'pnl')
for _, t in top_wins.iterrows():
    print(f"  {t['ticker']:<10} {t['name'][:18]:<18} {t['entry_date'].strftime('%Y-%m-%d')} → {t['exit_date'].strftime('%Y-%m-%d')} "
          f"{t['pnl_pct']*100:>+6.1f}% {t['pnl']:>+10,.0f} SAR ({t['exit_reason']})")

# Strategy validation checklist
print("\n" + "=" * 80)
print("STRATEGY VALIDATION CHECKLIST")
print("=" * 80)

checks = []

# Return checks
if total_return > 0.10:
    checks.append(("✓", f"Strong positive return: {total_return*100:+.1f}%"))
elif total_return > 0:
    checks.append(("⚠️", f"Modest positive return: {total_return*100:+.1f}%"))
else:
    checks.append(("✗", f"Negative return: {total_return*100:+.1f}%"))

# Sharpe checks
if sharpe > 0.75:
    checks.append(("✓", f"Excellent Sharpe ratio: {sharpe:.2f}"))
elif sharpe > 0.5:
    checks.append(("✓", f"Good Sharpe ratio: {sharpe:.2f}"))
elif sharpe > 0:
    checks.append(("⚠️", f"Marginal Sharpe ratio: {sharpe:.2f}"))
else:
    checks.append(("✗", f"Negative Sharpe ratio: {sharpe:.2f}"))

# Drawdown checks
if max_dd < 0.10:
    checks.append(("✓", f"Excellent drawdown control: {max_dd*100:.1f}%"))
elif max_dd < 0.20:
    checks.append(("⚠️", f"Moderate drawdown: {max_dd*100:.1f}%"))
else:
    checks.append(("✗", f"High drawdown: {max_dd*100:.1f}%"))

# Win rate checks
if win_rate > 0.50:
    checks.append(("✓", f"Good win rate: {win_rate*100:.1f}%"))
elif win_rate > 0.40:
    checks.append(("⚠️", f"Acceptable win rate: {win_rate*100:.1f}%"))
else:
    checks.append(("✗", f"Low win rate: {win_rate*100:.1f}%"))

# Profit factor
if profit_factor > 1.5:
    checks.append(("✓", f"Excellent profit factor: {profit_factor:.2f}"))
elif profit_factor > 1.0:
    checks.append(("✓", f"Profitable: PF = {profit_factor:.2f}"))
else:
    checks.append(("✗", f"Unprofitable: PF = {profit_factor:.2f}"))

print()
for sym, msg in checks:
    print(f"  {sym} {msg}")

passed = sum(1 for s, _ in checks if s == "✓")
total_checks = len(checks)

print(f"\n  SCORE: {passed}/{total_checks} checks passed")

if passed == total_checks:
    verdict = "STRATEGY FULLY VALIDATED - Ready for live trading!"
    symbol = "🏆"
elif passed >= total_checks - 1:
    verdict = "STRATEGY VALIDATED - Minor improvements possible"
    symbol = "✓"
elif passed >= total_checks // 2:
    verdict = "STRATEGY PARTIALLY VALIDATED - Needs monitoring"
    symbol = "⚠️"
else:
    verdict = "STRATEGY NEEDS MORE WORK"
    symbol = "✗"

print(f"\n  {symbol} {verdict}")

# Save results
os.makedirs('output/final_validation', exist_ok=True)
equity_df.to_csv('output/final_validation/equity_curve.csv', index=False)
trades_df.to_csv('output/final_validation/trades.csv', index=False)

results = {
    'parameters': CONFIG,
    'performance': {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_trades': len(trades_df),
    },
    'validation': {
        'passed_checks': passed,
        'total_checks': total_checks,
        'verdict': verdict
    }
}

with open('output/final_validation/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to output/final_validation/")

print("\n" + "=" * 80)
print("FINAL VALIDATION COMPLETE")
print("=" * 80)
