#!/usr/bin/env python3
"""
FINAL OPTIMIZED BACKTESTER
==========================
Key insight: Don't fight the market. Use signals for SELECTION, not timing.

Approach:
1. Quality stock selection (filter out weak stocks)
2. Momentum ranking (rotate to winners)
3. Modest leverage (1.5x max) only in confirmed bull
4. Let compounding work over time
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


# High-quality TASI stocks (blue chips + growth)
TASI_TICKERS = [
    # Banks (high quality)
    '1180.SR', '1010.SR', '1150.SR', '1140.SR',
    # Energy
    '2222.SR', '2082.SR',
    # Petrochemicals
    '2010.SR', '2020.SR', '2350.SR',
    # Materials
    '1211.SR', '1304.SR',
    # Retail/Consumer
    '4190.SR', '4001.SR', '4007.SR',
    # Telecom
    '7010.SR', '7020.SR',
    # Insurance
    '8010.SR', '8210.SR',
    # Real Estate
    '4300.SR',
    # Diversified
    '4280.SR',
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
    print(f"Loaded {len(data)} high-quality stocks")
    return data


def quality_momentum_score(df, idx):
    """Score based on quality AND momentum."""
    if idx < 252:
        return -999
    
    c = df['close']
    v = df['volume']
    
    # Momentum (multiple timeframes)
    m1 = c.iloc[idx] / c.iloc[idx-21] - 1
    m3 = c.iloc[idx] / c.iloc[idx-63] - 1
    m6 = c.iloc[idx] / c.iloc[idx-126] - 1
    m12 = c.iloc[idx] / c.iloc[idx-252] - 1
    
    # Quality filters
    ma50 = c.iloc[idx-50:idx].mean()
    ma200 = c.iloc[idx-200:idx].mean()
    
    # Must be above 200MA (quality filter)
    if c.iloc[idx] < ma200 * 0.95:
        return -999
    
    # Volatility (prefer lower)
    vol = c.pct_change().iloc[idx-60:idx].std()
    vol_penalty = vol * 2  # Penalize high vol
    
    # Momentum score
    mom = m1 * 0.3 + m3 * 0.3 + m6 * 0.25 + m12 * 0.15
    
    # Trend bonus
    trend_bonus = 0
    if c.iloc[idx] > ma50:
        trend_bonus += 0.05
    if ma50 > ma200:
        trend_bonus += 0.05
    
    return mom + trend_bonus - vol_penalty


def run():
    print("=" * 80)
    print("📈 FINAL OPTIMIZED BACKTESTER")
    print("Using signals for SELECTION, not timing")
    print("=" * 80)
    
    data = fetch_data()
    
    all_dates = set.intersection(*[set(df.index) for df in data.values()])
    dates = sorted(list(all_dates))
    start_idx = 252
    
    market = data.get('2222.SR', list(data.values())[0])
    
    strategies = {}
    
    # =========================================================================
    # STRATEGY 1: Quality Momentum Rotation (Weekly)
    # =========================================================================
    print("\nStrategy 1: Quality Momentum Rotation (Weekly rebalance)...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Rebalance weekly
        should_rebalance = (i - start_idx) % 5 == 0
        
        if should_rebalance or i == start_idx:
            # Score all stocks
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            # Top half of qualifying stocks
            n_hold = max(5, len([s for s in scores if s[1] > 0]) // 2)
            holdings = [s[0] for s in scores[:n_hold]]
        
        # Calculate return
        ret = 0
        for t in holdings:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r / len(holdings)
        
        portfolio.append(portfolio[-1] * (1 + ret))
    
    strategies['Quality Momentum (Weekly)'] = portfolio
    
    # =========================================================================
    # STRATEGY 2: Momentum Rotation with Mild Leverage
    # =========================================================================
    print("Strategy 2: Momentum + Mild Leverage (1.5x bull)...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Market regime (simple)
        if d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 50:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                leverage = 1.5 if p > ma50 else 1.0
            else:
                leverage = 1.0
        else:
            leverage = 1.0
        
        # Rebalance weekly
        if (i - start_idx) % 5 == 0 or i == start_idx:
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            holdings = [s[0] for s in scores[:8] if s[1] > 0]
            if len(holdings) < 5:
                holdings = [s[0] for s in scores[:5]]
        
        ret = 0
        for t in holdings:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r / len(holdings)
        
        ret *= leverage
        portfolio.append(portfolio[-1] * (1 + ret))
    
    strategies['Momentum + 1.5x Leverage'] = portfolio
    
    # =========================================================================
    # STRATEGY 3: Concentrated Quality (Top 5)
    # =========================================================================
    print("Strategy 3: Concentrated Quality (Top 5 only)...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Monthly rebalance
        if (i - start_idx) % 21 == 0 or i == start_idx:
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            holdings = [s[0] for s in scores[:5]]
        
        ret = 0
        for t in holdings:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r / len(holdings)
        
        portfolio.append(portfolio[-1] * (1 + ret))
    
    strategies['Concentrated (Top 5)'] = portfolio
    
    # =========================================================================
    # STRATEGY 4: Adaptive (More stocks in bull, fewer in bear)
    # =========================================================================
    print("Strategy 4: Adaptive Position Count...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Market regime determines # of holdings
        if d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 200:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                ma200 = market['close'].iloc[idx-200:idx].mean()
                
                if p > ma50 > ma200:
                    n_hold = 10  # Strong bull - more diversified
                    leverage = 1.3
                elif p > ma50:
                    n_hold = 7
                    leverage = 1.0
                elif p > ma200:
                    n_hold = 5
                    leverage = 0.8
                else:
                    n_hold = 3  # Bear - very concentrated, defensive
                    leverage = 0.5
            else:
                n_hold = 7
                leverage = 1.0
        else:
            n_hold = 7
            leverage = 1.0
        
        # Rebalance weekly
        if (i - start_idx) % 5 == 0 or i == start_idx:
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            holdings = [s[0] for s in scores[:n_hold]]
        
        ret = 0
        for t in holdings:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r / len(holdings)
        
        ret *= leverage
        portfolio.append(portfolio[-1] * (1 + ret))
    
    strategies['Adaptive'] = portfolio
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    print("Baseline: Equal Weight Buy & Hold...")
    
    portfolio = [100]
    all_stocks = list(data.keys())
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        ret = 0
        count = 0
        for t in all_stocks:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r
                count += 1
        
        if count > 0:
            ret /= count
        
        portfolio.append(portfolio[-1] * (1 + ret))
    
    strategies['Buy & Hold (Equal)'] = portfolio
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 110)
    print("📊 FINAL BACKTEST RESULTS (2019-2026)")
    print("=" * 110)
    
    baseline = (strategies['Buy & Hold (Equal)'][-1] / 100 - 1) * 100
    
    results = []
    for name, eq in strategies.items():
        total = (eq[-1] / 100 - 1) * 100
        yrs = len(eq) / 252
        annual = ((eq[-1] / 100) ** (1/yrs) - 1) * 100 if yrs > 0 else 0
        
        rets = np.diff(eq) / np.array(eq[:-1])
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        
        rm = np.maximum.accumulate(eq)
        dd = (rm - eq) / rm * 100
        mdd = np.max(dd)
        
        calmar = annual / mdd if mdd > 0 else 0
        excess = total - baseline
        
        results.append({
            'name': name, 'total': total, 'annual': annual,
            'sharpe': sharpe, 'calmar': calmar, 'mdd': mdd, 'excess': excess
        })
    
    results.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n  {'Strategy':<35} {'Total%':>10} {'Annual%':>10} {'Sharpe':>8} {'Calmar':>8} {'MaxDD%':>8} {'Excess':>10}")
    print("  " + "-" * 100)
    
    for r in results:
        emoji = "🏆" if r['excess'] > 20 else "🥇" if r['excess'] > 10 else "✅" if r['excess'] > 0 else "❌"
        print(f"  {emoji} {r['name']:<33} {r['total']:>+9.1f}% {r['annual']:>+9.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['mdd']:>7.1f}% {r['excess']:>+9.1f}%")
    
    w = results[0]
    
    print(f"""

{'='*110}
🏆 FINAL CHAMPION: {w['name']}
{'='*110}

  PERFORMANCE (2019-2026):
  ─────────────────────────────────────────────────────────────────────
    Total Return:        {w['total']:+.1f}%
    Annual Return:       {w['annual']:+.1f}%
    Sharpe Ratio:        {w['sharpe']:.2f}
    Calmar Ratio:        {w['calmar']:.2f}
    Max Drawdown:        {w['mdd']:.1f}%
  
  VS BUY & HOLD:
  ─────────────────────────────────────────────────────────────────────
    Excess Return:       {w['excess']:+.1f}%
    Outperformance:      {(w['total']+100)/(baseline+100):.2f}x

  KEY SUCCESS FACTORS:
  ─────────────────────────────────────────────────────────────────────
    1. Quality stock universe (blue chips + growth)
    2. Momentum-based rotation (ride winners)
    3. Simple regime filter (not over-leveraged)
    4. Weekly rebalancing (capture trends without over-trading)
    5. Let compounding work

{'='*110}
""")
    
    return results


if __name__ == "__main__":
    run()
