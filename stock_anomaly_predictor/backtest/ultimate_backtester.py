#!/usr/bin/env python3
"""
ULTIMATE STRATEGY BACKTESTER
============================
Maximum aggression when conditions align:
- 5x leverage in strong bull
- 0x (cash) in bear
- Best 3 momentum stocks only
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


TASI_TICKERS = [
    '1180.SR', '1010.SR', '1150.SR', '1140.SR', '1060.SR', '1050.SR',
    '1020.SR', '2222.SR', '4030.SR', '2010.SR', '2350.SR', '2020.SR',
    '1211.SR', '1321.SR', '1304.SR', '2240.SR', '2320.SR', '2082.SR',
    '4190.SR', '4003.SR', '4001.SR', '4007.SR', '4280.SR', '2280.SR',
    '4071.SR', '7010.SR', '7020.SR', '4300.SR', '8010.SR', '8210.SR',
]


def fetch_data(start: str = '2019-01-01'):
    print(f"Fetching data from {start}...")
    data = {}
    for t in TASI_TICKERS:
        try:
            df = yf.Ticker(t).history(start=start)
            if len(df) >= 252:
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)
                data[t] = df
        except:
            pass
    print(f"Loaded {len(data)} stocks")
    return data


def momentum_score(df, idx):
    if idx < 252:
        return -999
    c = df['close']
    m1 = c.iloc[idx] / c.iloc[idx-21] - 1
    m3 = c.iloc[idx] / c.iloc[idx-63] - 1
    m6 = c.iloc[idx] / c.iloc[idx-126] - 1
    m12 = c.iloc[idx] / c.iloc[idx-252] - 1
    
    # Skip stocks in downtrend
    ma50 = c.iloc[idx-50:idx].mean()
    ma200 = c.iloc[idx-200:idx].mean()
    if c.iloc[idx] < ma200:
        return -999
    
    return m1 * 0.4 + m3 * 0.3 + m6 * 0.2 + m12 * 0.1


def run():
    print("=" * 80)
    print("🚀 ULTIMATE STRATEGY BACKTESTER")
    print("=" * 80)
    
    data = fetch_data()
    
    all_dates = set.intersection(*[set(df.index) for df in data.values()])
    dates = sorted(list(all_dates))
    start_idx = 252
    
    market = data.get('2222.SR', list(data.values())[0])
    
    # Different leverage configurations
    configs = [
        ("Conservative (2x bull/0.5x bear)", 2.0, 0.5, 5),
        ("Moderate (3x bull/0.3x bear)", 3.0, 0.3, 4),
        ("Aggressive (4x bull/0x bear)", 4.0, 0.0, 3),
        ("Ultra (5x bull/0x bear)", 5.0, 0.0, 3),
        ("Max (6x bull/0x bear)", 6.0, 0.0, 2),
    ]
    
    results = {}
    
    for name, bull_lev, bear_lev, n_stocks in configs:
        print(f"\nRunning {name}...")
        
        portfolio = [100]
        peak = 100
        
        for i in range(start_idx, len(dates) - 1):
            d = dates[i]
            d_next = dates[i + 1]
            
            # Market regime
            if d in market.index:
                idx = market.index.get_loc(d)
                if idx >= 200:
                    p = market['close'].iloc[idx]
                    ma50 = market['close'].iloc[idx-50:idx].mean()
                    ma200 = market['close'].iloc[idx-200:idx].mean()
                    
                    if p > ma50 > ma200:
                        regime = 'BULL'
                        leverage = bull_lev
                    elif p > ma50:
                        regime = 'WEAK_BULL'
                        leverage = bull_lev * 0.5
                    elif p > ma200:
                        regime = 'NEUTRAL'
                        leverage = 1.0
                    else:
                        regime = 'BEAR'
                        leverage = bear_lev
                else:
                    leverage = 1.0
            else:
                leverage = 1.0
            
            # Drawdown protection
            curr_dd = (peak - portfolio[-1]) / peak if peak > 0 else 0
            if curr_dd > 0.25:
                leverage = min(1.0, leverage)
            elif curr_dd > 0.15:
                leverage *= 0.5
            
            # Top momentum stocks
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top = [s[0] for s in scores[:n_stocks] if s[1] > 0]
            
            if not top:
                top = [s[0] for s in scores[:2]]
            
            # Calculate return
            ret = 0
            for t in top:
                df = data[t]
                if d in df.index and d_next in df.index:
                    r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                    ret += r / len(top)
            
            ret *= leverage
            ret = max(-0.15, min(0.20, ret))  # Cap extreme moves
            
            new_val = portfolio[-1] * (1 + ret)
            portfolio.append(new_val)
            peak = max(peak, new_val)
        
        results[name] = portfolio
    
    # Baseline
    print("\nRunning Baseline...")
    portfolio = [100]
    bl_stocks = list(data.keys())[:15]
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        rets = []
        for t in bl_stocks:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                rets.append(r)
        
        if rets:
            portfolio.append(portfolio[-1] * (1 + np.mean(rets)))
        else:
            portfolio.append(portfolio[-1])
    
    results['Buy & Hold'] = portfolio
    
    # Print results
    print("\n" + "=" * 110)
    print("🏆 ULTIMATE BACKTEST RESULTS (2019-2026)")
    print("=" * 110)
    
    baseline = (results['Buy & Hold'][-1] / 100 - 1) * 100
    
    summary = []
    for name, eq in results.items():
        total = (eq[-1] / 100 - 1) * 100
        yrs = len(eq) / 252
        annual = ((eq[-1] / 100) ** (1/yrs) - 1) * 100 if yrs > 0 else 0
        rets = np.diff(eq) / np.array(eq[:-1])
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        rm = np.maximum.accumulate(eq)
        dd = (rm - eq) / rm * 100
        mdd = np.max(dd)
        excess = total - baseline
        
        summary.append({
            'name': name, 'total': total, 'annual': annual,
            'sharpe': sharpe, 'mdd': mdd, 'excess': excess
        })
    
    summary.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n  {'Strategy':<40} {'Total%':>10} {'Annual%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Excess':>10}")
    print("  " + "-" * 95)
    
    for s in summary:
        emoji = "🏆" if s['excess'] > 100 else "🥇" if s['excess'] > 50 else "🥈" if s['excess'] > 25 else "✅" if s['excess'] > 0 else "❌"
        print(f"  {emoji} {s['name']:<38} {s['total']:>+9.1f}% {s['annual']:>+9.1f}% "
              f"{s['sharpe']:>7.2f} {s['mdd']:>7.1f}% {s['excess']:>+9.1f}%")
    
    w = summary[0]
    print(f"""

{'='*110}
🏆🏆🏆 ULTIMATE CHAMPION: {w['name']}
{'='*110}

  Total Return:       {w['total']:+.1f}%
  Annual Return:      {w['annual']:+.1f}%
  Sharpe Ratio:       {w['sharpe']:.2f}
  Max Drawdown:       {w['mdd']:.1f}%
  vs Buy & Hold:      {w['excess']:+.1f}%
  Multiplier:         {(w['total']+100)/(baseline+100):.2f}x

{'='*110}
""")

if __name__ == "__main__":
    run()
