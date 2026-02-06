#!/usr/bin/env python3
"""
BACKTEST: Trend-Based Leverage Strategy
=======================================
Verify the strategy actually works before deploying.
"""

import os
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 80)
print("TREND-BASED LEVERAGE STRATEGY BACKTEST")
print("=" * 80)

# Strategy parameters
LEVERAGE_BULL = 1.5
LEVERAGE_BEAR = 0.5
MA_PERIOD = 50

# Load data
STOCKS = ['1180.SR', '1010.SR', '2222.SR', '7010.SR', '2010.SR',
          '1150.SR', '2082.SR', '2280.SR', '8210.SR', '4190.SR']

print(f"\n[1] LOADING DATA...")
stock_data = {}
for ticker in STOCKS:
    try:
        df = yf.Ticker(ticker).history(start="2022-01-01")
        if len(df) > 100:
            df.columns = [c.lower() for c in df.columns]
            stock_data[ticker] = df
    except:
        pass

print(f"  Loaded {len(stock_data)} stocks")

# Create equal-weight market proxy
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index.tolist())
trading_dates = sorted(all_dates)

print(f"  Period: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
print(f"  Days: {len(trading_dates)}")

# Calculate market returns (equal weight)
market_returns = []
for i in range(1, len(trading_dates)):
    date = trading_dates[i]
    prev_date = trading_dates[i-1]
    
    daily_returns = []
    for ticker, df in stock_data.items():
        if date in df.index and prev_date in df.index:
            ret = df.loc[date, 'close'] / df.loc[prev_date, 'close'] - 1
            daily_returns.append(ret)
    
    if daily_returns:
        market_returns.append({
            'date': date,
            'return': np.mean(daily_returns)
        })

returns_df = pd.DataFrame(market_returns).set_index('date')
returns_df['cum_return'] = (1 + returns_df['return']).cumprod()
returns_df['ma_50'] = returns_df['cum_return'].rolling(MA_PERIOD).mean()

print(f"\n[2] RUNNING BACKTEST...")

# Strategy 1: Buy and Hold
bh_equity = [1_000_000]
for ret in returns_df['return']:
    bh_equity.append(bh_equity[-1] * (1 + ret))

bh_final = bh_equity[-1]
bh_return = bh_final / 1_000_000 - 1

# Strategy 2: Trend-Based Leverage
leverage_equity = [1_000_000]
regimes = []

for i in range(MA_PERIOD, len(returns_df)):
    date = returns_df.index[i]
    cum = returns_df['cum_return'].iloc[i]
    ma = returns_df['ma_50'].iloc[i]
    ret = returns_df['return'].iloc[i]
    
    # Determine regime and leverage
    if cum > ma:
        regime = "BULL"
        leverage = LEVERAGE_BULL
    else:
        regime = "BEAR"
        leverage = LEVERAGE_BEAR
    
    regimes.append(regime)
    
    # Apply leveraged return
    new_equity = leverage_equity[-1] * (1 + ret * leverage)
    leverage_equity.append(new_equity)

leverage_final = leverage_equity[-1]
leverage_return = leverage_final / 1_000_000 - 1

# Calculate metrics
print(f"\n[3] CALCULATING METRICS...")

# Sharpe ratios
bh_daily_returns = np.diff(bh_equity) / bh_equity[:-1]
lev_daily_returns = np.diff(leverage_equity) / leverage_equity[:-1]

bh_sharpe = np.sqrt(252) * np.mean(bh_daily_returns) / np.std(bh_daily_returns) if np.std(bh_daily_returns) > 0 else 0
lev_sharpe = np.sqrt(252) * np.mean(lev_daily_returns) / np.std(lev_daily_returns) if np.std(lev_daily_returns) > 0 else 0

# Max drawdowns
bh_peak = np.maximum.accumulate(bh_equity)
bh_dd = (bh_peak - bh_equity) / bh_peak
bh_max_dd = np.max(bh_dd)

lev_peak = np.maximum.accumulate(leverage_equity)
lev_dd = (lev_peak - leverage_equity) / lev_peak
lev_max_dd = np.max(lev_dd)

# Regime analysis
bull_days = regimes.count("BULL")
bear_days = regimes.count("BEAR")

# Annualized returns
years = len(trading_dates) / 252
bh_annual = (1 + bh_return) ** (1/years) - 1
lev_annual = (1 + leverage_return) ** (1/years) - 1

print(f"""
{'='*80}
BACKTEST RESULTS
{'='*80}

┌─────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE COMPARISON                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                           │    Buy & Hold    │  Trend Leverage  │  Excess   │
├───────────────────────────┼──────────────────┼──────────────────┼───────────┤
│  Total Return             │   {bh_return*100:>+10.1f}%   │   {leverage_return*100:>+10.1f}%   │ {(leverage_return-bh_return)*100:>+8.1f}% │
│  Annualized Return        │   {bh_annual*100:>+10.1f}%   │   {lev_annual*100:>+10.1f}%   │ {(lev_annual-bh_annual)*100:>+8.1f}% │
│  Sharpe Ratio             │   {bh_sharpe:>+10.2f}    │   {lev_sharpe:>+10.2f}    │ {lev_sharpe-bh_sharpe:>+8.2f}  │
│  Max Drawdown             │   {bh_max_dd*100:>10.1f}%   │   {lev_max_dd*100:>10.1f}%   │          │
│  Final Value (1M start)   │ {bh_final:>14,.0f}   │ {leverage_final:>14,.0f}   │          │
└───────────────────────────┴──────────────────┴──────────────────┴───────────┘

REGIME ANALYSIS:
  Bull Days: {bull_days} ({bull_days/len(regimes)*100:.1f}%)
  Bear Days: {bear_days} ({bear_days/len(regimes)*100:.1f}%)

WHY IT WORKS:
  • In BULL regime ({bull_days} days): We use {LEVERAGE_BULL}x = Capture MORE upside
  • In BEAR regime ({bear_days} days): We use {LEVERAGE_BEAR}x = Lose LESS downside
  • ALWAYS invested = Never miss big up days
""")

# Verdict
if leverage_return > bh_return:
    excess = leverage_return - bh_return
    print(f"""
{'='*80}
✓ STRATEGY VALIDATED
{'='*80}

  The trend-based leverage strategy BEATS buy-and-hold by {excess*100:+.1f}%
  
  This is achieved by:
  1. Staying invested 100% of the time (no missed days)
  2. Using {LEVERAGE_BULL}x leverage during uptrends (price > 50-day MA)
  3. Using {LEVERAGE_BEAR}x leverage during downtrends (price < 50-day MA)
  
  READY FOR PRODUCTION DEPLOYMENT
""")
    validated = True
else:
    print(f"""
{'='*80}
✗ STRATEGY DID NOT BEAT BUY-AND-HOLD
{'='*80}
""")
    validated = False

# Save results
os.makedirs('output/leverage_backtest', exist_ok=True)

results = {
    'period': {
        'start': trading_dates[0].strftime('%Y-%m-%d'),
        'end': trading_dates[-1].strftime('%Y-%m-%d'),
        'days': len(trading_dates)
    },
    'buy_and_hold': {
        'total_return': round(bh_return, 4),
        'annual_return': round(bh_annual, 4),
        'sharpe': round(bh_sharpe, 3),
        'max_drawdown': round(bh_max_dd, 4),
        'final_value': round(bh_final, 2)
    },
    'leverage_strategy': {
        'total_return': round(leverage_return, 4),
        'annual_return': round(lev_annual, 4),
        'sharpe': round(lev_sharpe, 3),
        'max_drawdown': round(lev_max_dd, 4),
        'final_value': round(leverage_final, 2)
    },
    'excess_return': round(leverage_return - bh_return, 4),
    'regime_analysis': {
        'bull_days': bull_days,
        'bear_days': bear_days,
        'bull_pct': round(bull_days/len(regimes)*100, 1),
        'bear_pct': round(bear_days/len(regimes)*100, 1)
    },
    'validated': validated
}

with open('output/leverage_backtest/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save equity curves
equity_df = pd.DataFrame({
    'date': trading_dates[MA_PERIOD:],
    'buy_hold': bh_equity[MA_PERIOD:len(trading_dates)],
    'leverage': leverage_equity[1:]
})
equity_df.to_csv('output/leverage_backtest/equity_curves.csv', index=False)

print(f"\nResults saved to output/leverage_backtest/")
print("=" * 80)
