#!/usr/bin/env python3
"""
ULTRA-FAST EXIT OPTIMIZATION
============================
Vectorized, minimal version for quick results.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except:
    import os
    os.system("pip install yfinance pandas numpy --quiet")
    import yfinance as yf

print("=" * 100)
print("🏆 EXIT STRATEGY OPTIMIZATION - 4 TEAMS")
print("=" * 100)

# Minimal stock list for speed
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
           'JPM', 'V', 'MA', 'UNH', 'JNJ', 'LLY', 'WMT', 'PG', 'HD', 'CAT',
           'XOM', 'CVX', 'NFLX', 'LIN', 'NEE', 'CRM', 'AMD', 'COST']

print(f"\nFetching {len(TICKERS)} stocks...")

# Fetch data
data = {}
for t in TICKERS:
    try:
        df = yf.Ticker(t).history(start='2021-01-01')
        if len(df) > 252:
            df.columns = [c.lower() for c in df.columns]
            df.index = df.index.tz_localize(None)
            data[t] = df['close']
    except:
        pass

print(f"Loaded {len(data)} stocks")

# Get market
spy = yf.Ticker('SPY').history(start='2021-01-01')
spy.columns = [c.lower() for c in spy.columns]
spy.index = spy.index.tz_localize(None)
market = spy['close']

# Align dates
prices = pd.DataFrame(data)
prices = prices.dropna()
market = market.reindex(prices.index).ffill()

print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Trading days: {len(prices)}")

# Pre-compute returns and momentum scores
returns = prices.pct_change()
mom_1m = prices.pct_change(21)
mom_3m = prices.pct_change(63)
ma_200 = prices.rolling(200).mean()


def get_top_stocks(date_idx, n):
    """Get top n momentum stocks."""
    if date_idx < 200:
        return list(prices.columns[:n])
    
    # Momentum score
    m1 = mom_1m.iloc[date_idx]
    m3 = mom_3m.iloc[date_idx]
    score = m1 * 0.5 + m3 * 0.5
    
    # Quality filter
    price = prices.iloc[date_idx]
    ma = ma_200.iloc[date_idx]
    mask = price > ma * 0.95
    score = score[mask]
    
    return score.nlargest(n).index.tolist()


def simulate(n_stocks, rebalance_days, use_leverage, stop_loss, take_profit, trailing_stop):
    """Fast simulation."""
    
    portfolio = 100.0
    peak = 100.0
    max_dd = 0.0
    daily_rets = []
    
    positions = {}  # {ticker: {'entry': price, 'peak': price}}
    
    start = 252
    
    for i in range(start, len(prices) - 1):
        # Leverage
        lev = 1.0
        if use_leverage and i >= 50:
            if market.iloc[i] > market.iloc[i-50:i].mean():
                lev = 1.5
        
        # Check exits
        to_remove = []
        for t, pos in positions.items():
            curr = prices[t].iloc[i]
            entry = pos['entry']
            pk = pos['peak']
            
            # Update peak
            if curr > pk:
                positions[t]['peak'] = curr
                pk = curr
            
            # Stop loss
            if stop_loss and (curr - entry) / entry <= -stop_loss:
                to_remove.append(t)
                continue
            
            # Take profit
            if take_profit and (curr - entry) / entry >= take_profit:
                to_remove.append(t)
                continue
            
            # Trailing stop
            if trailing_stop and (curr - pk) / pk <= -trailing_stop:
                to_remove.append(t)
                continue
        
        for t in to_remove:
            del positions[t]
        
        # Rebalance
        if (i - start) % rebalance_days == 0:
            top = get_top_stocks(i, n_stocks)
            
            # Sell not in top
            for t in list(positions.keys()):
                if t not in top:
                    del positions[t]
            
            # Buy new
            for t in top:
                if t not in positions:
                    positions[t] = {'entry': prices[t].iloc[i], 'peak': prices[t].iloc[i]}
        
        # Daily return
        day_ret = 0.0
        if positions:
            for t in positions:
                r = returns[t].iloc[i + 1]
                day_ret += r / len(positions)
        
        day_ret *= lev
        daily_rets.append(day_ret)
        
        portfolio *= (1 + day_ret)
        if portfolio > peak:
            peak = portfolio
        dd = (portfolio - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    total = portfolio / 100 - 1
    arr = np.array(daily_rets)
    sharpe = arr.mean() / arr.std() * np.sqrt(252) if arr.std() > 0 else 0
    
    return total, sharpe, max_dd


def optimize_strategy(name, n_stocks, rebalance, leverage):
    """Optimize one strategy."""
    
    print(f"\n{'='*80}")
    print(f"🔬 {name}")
    print(f"{'='*80}")
    
    # Baseline
    base_ret, base_sharpe, base_dd = simulate(n_stocks, rebalance, leverage, None, None, None)
    print(f"\n📊 BASELINE: Return={base_ret*100:+.1f}%, Sharpe={base_sharpe:.2f}, MaxDD={base_dd*100:.1f}%")
    
    # Parameters to test
    sls = [None, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    tps = [None, 0.15, 0.25, 0.40, 0.60, 1.00]
    trails = [None, 0.08, 0.12, 0.15, 0.20]
    
    results = []
    total = len(sls) * len(tps) * len(trails)
    print(f"Testing {total} combinations...")
    
    for sl in sls:
        for tp in tps:
            for ts in trails:
                ret, sharpe, dd = simulate(n_stocks, rebalance, leverage, sl, tp, ts)
                results.append({
                    'sl': sl, 'tp': tp, 'ts': ts,
                    'ret': ret, 'sharpe': sharpe, 'dd': dd,
                    'vs_base': ret - base_ret
                })
    
    # Sort by return
    results.sort(key=lambda x: x['ret'], reverse=True)
    
    return results, base_ret, base_sharpe, base_dd


# Run all 4 strategies
strategies = {
    'TEAM 1 - Concentrated (Top 5)': (5, 21, False),
    'TEAM 2 - Momentum 1.5x': (12, 5, True),
    'TEAM 3 - Quality Momentum': (15, 5, False),
    'TEAM 4 - Adaptive': (10, 5, False),
}

all_results = {}
for name, (n, rebal, lev) in strategies.items():
    results, base_ret, base_sharpe, base_dd = optimize_strategy(name, n, rebal, lev)
    all_results[name] = {
        'results': results,
        'baseline': {'ret': base_ret, 'sharpe': base_sharpe, 'dd': base_dd}
    }

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 100)
print("📊 OPTIMIZATION RESULTS")
print("=" * 100)

for name, d in all_results.items():
    res = d['results']
    base = d['baseline']
    
    print(f"\n{'─'*100}")
    print(f"📈 {name}")
    print(f"{'─'*100}")
    print(f"  BASELINE: {base['ret']*100:+.1f}% return | {base['sharpe']:.2f} Sharpe | {base['dd']*100:.1f}% MaxDD")
    
    print(f"\n  🏆 TOP 10 BY RETURN:")
    print(f"  {'Rank':<6} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10} {'vs Base':<10}")
    print(f"  {'-'*90}")
    
    for i, r in enumerate(res[:10]):
        sl = f"{r['sl']*100:.0f}%" if r['sl'] else "None"
        tp = f"{r['tp']*100:.0f}%" if r['tp'] else "None"
        ts = f"{r['ts']*100:.0f}%" if r['ts'] else "None"
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        print(f"  {rank:<6} {sl:<12} {tp:<14} {ts:<12} {r['ret']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['dd']*100:>9.1f}% {r['vs_base']*100:>+8.1f}%")
    
    # Best Sharpe
    by_sharpe = sorted(res, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  📊 TOP 5 BY SHARPE:")
    for i, r in enumerate(by_sharpe[:5]):
        sl = f"{r['sl']*100:.0f}%" if r['sl'] else "None"
        tp = f"{r['tp']*100:.0f}%" if r['tp'] else "None"
        ts = f"{r['ts']*100:.0f}%" if r['ts'] else "None"
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        print(f"  {rank} SL={sl:<8} TP={tp:<10} Trail={ts:<8} → {r['ret']*100:+.1f}% ret, {r['sharpe']:.2f} Sharpe")
    
    # Lowest DD
    profitable = [r for r in res if r['ret'] > 0]
    if profitable:
        by_dd = sorted(profitable, key=lambda x: x['dd'], reverse=True)
        print(f"\n  🛡️ TOP 5 LOWEST DRAWDOWN (Profitable):")
        for i, r in enumerate(by_dd[:5]):
            sl = f"{r['sl']*100:.0f}%" if r['sl'] else "None"
            tp = f"{r['tp']*100:.0f}%" if r['tp'] else "None"
            ts = f"{r['ts']*100:.0f}%" if r['ts'] else "None"
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            print(f"  {rank} SL={sl:<8} TP={tp:<10} Trail={ts:<8} → {r['ret']*100:+.1f}% ret, {r['dd']*100:.1f}% DD")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("🏆 OPTIMAL EXIT PARAMETERS")
print("=" * 100)

print(f"\n  {'Strategy':<35} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Δ Return':<10}")
print(f"  {'-'*95}")

for name, d in all_results.items():
    best = d['results'][0]
    base = d['baseline']
    
    sl = f"{best['sl']*100:.0f}%" if best['sl'] else "None"
    tp = f"{best['tp']*100:.0f}%" if best['tp'] else "None"
    ts = f"{best['ts']*100:.0f}%" if best['ts'] else "None"
    delta = best['ret'] - base['ret']
    
    emoji = "🚀" if delta > 0.1 else "✅" if delta > 0 else "➖" if delta > -0.05 else "⚠️"
    print(f"  {emoji} {name:<33} {sl:<12} {tp:<14} {ts:<12} {best['ret']*100:>+10.1f}% {delta*100:>+8.1f}%")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 100)
print("💡 FINAL VERDICT PER TEAM")
print("=" * 100)

for name, d in all_results.items():
    best = d['results'][0]
    base = d['baseline']
    delta = best['ret'] - base['ret']
    
    sl = f"{best['sl']*100:.0f}%" if best['sl'] else "None"
    tp = f"{best['tp']*100:.0f}%" if best['tp'] else "None"
    ts = f"{best['ts']*100:.0f}%" if best['ts'] else "None"
    
    # Count improvements
    improved = sum(1 for r in d['results'] if r['ret'] > base['ret'])
    total = len(d['results'])
    
    if delta > 0.05:
        verdict = "EXITS SIGNIFICANTLY HELP"
        action = f"USE: SL={sl}, TP={tp}, Trail={ts}"
    elif delta > 0:
        verdict = "EXITS MARGINALLY HELP"
        action = f"OPTIONAL: SL={sl}, TP={tp}, Trail={ts}"
    elif delta > -0.02:
        verdict = "EXITS NEUTRAL"
        action = "EITHER WAY WORKS"
    else:
        verdict = "EXITS HURT PERFORMANCE"
        action = "KEEP NO EXITS (BASELINE)"
    
    print(f"""
  📌 {name}
     Baseline: {base['ret']*100:+.1f}% | Best: {best['ret']*100:+.1f}% | Δ: {delta*100:+.1f}%
     Configs that beat baseline: {improved}/{total} ({improved/total*100:.0f}%)
     
     VERDICT: {verdict}
     ACTION: {action}
""")

print("=" * 100)
print("END OF OPTIMIZATION")
print("=" * 100)
