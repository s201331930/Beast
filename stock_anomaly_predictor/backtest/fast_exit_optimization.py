#!/usr/bin/env python3
"""
FAST EXIT STRATEGY OPTIMIZATION
================================
Optimized version with parallel-like efficiency and reduced parameter space.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


# Reduced stock universe for speed
US_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
    'ADBE', 'CRM', 'AMD', 'QCOM', 'NOW', 'INTU',
    'JPM', 'V', 'MA', 'BAC', 'GS', 'BLK',
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO',
    'WMT', 'PG', 'KO', 'COST', 'HD', 'MCD',
    'CAT', 'DE', 'HON', 'RTX', 'LMT', 'GE',
    'XOM', 'CVX', 'COP', 'SLB',
    'NFLX', 'DIS', 'CMCSA',
    'LIN', 'APD', 'SHW',
    'NEE', 'DUK',
]

print("Loading data...")
DATA_CACHE = {}
MARKET_CACHE = None


def load_all_data():
    global DATA_CACHE, MARKET_CACHE
    
    for t in US_TICKERS:
        try:
            df = yf.Ticker(t).history(start='2020-01-01')
            if len(df) >= 252:
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)
                DATA_CACHE[t] = df
        except:
            pass
    
    df = yf.Ticker('SPY').history(start='2020-01-01')
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    MARKET_CACHE = df
    
    print(f"Loaded {len(DATA_CACHE)} stocks")


def momentum_score(df, idx):
    if idx < 252:
        return -999
    c = df['close']
    m1 = c.iloc[idx] / c.iloc[idx-21] - 1
    m3 = c.iloc[idx] / c.iloc[idx-63] - 1
    m6 = c.iloc[idx] / c.iloc[idx-126] - 1
    
    ma200 = c.iloc[idx-200:idx].mean()
    if c.iloc[idx] < ma200 * 0.95:
        return -999
    
    return m1 * 0.4 + m3 * 0.35 + m6 * 0.25


def run_strategy(n_stocks: int, rebalance_freq: int, use_leverage: bool,
                 stop_loss: Optional[float], take_profit: Optional[float],
                 trailing_stop: Optional[float]) -> Dict:
    """Run single strategy configuration."""
    
    data = DATA_CACHE
    market = MARKET_CACHE
    
    all_dates = set(market.index)
    for df in data.values():
        all_dates &= set(df.index)
    dates = sorted(list(all_dates))
    start_idx = 252
    
    portfolio = 100.0
    peak_portfolio = 100.0
    max_dd = 0.0
    daily_returns = []
    
    positions = {}  # {ticker: {'entry': price, 'peak': price}}
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Leverage
        leverage = 1.0
        if use_leverage and d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 50:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                leverage = 1.5 if p > ma50 else 1.0
        
        # Check exits
        to_sell = []
        for ticker, pos in positions.items():
            if ticker not in data or d not in data[ticker].index:
                to_sell.append(ticker)
                continue
            
            price = data[ticker].loc[d, 'close']
            entry = pos['entry']
            peak = pos['peak']
            
            # Update peak
            if price > peak:
                positions[ticker]['peak'] = price
                peak = price
            
            # Stop-loss
            if stop_loss and (price - entry) / entry <= -stop_loss:
                to_sell.append(ticker)
                continue
            
            # Take-profit
            if take_profit and (price - entry) / entry >= take_profit:
                to_sell.append(ticker)
                continue
            
            # Trailing stop
            if trailing_stop and (price - peak) / peak <= -trailing_stop:
                to_sell.append(ticker)
                continue
        
        for t in to_sell:
            if t in positions:
                del positions[t]
        
        # Rebalance
        if (i - start_idx) % rebalance_freq == 0 or i == start_idx:
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            new_holdings = [s[0] for s in scores[:n_stocks]]
            
            # Sell not in new
            for t in list(positions.keys()):
                if t not in new_holdings:
                    del positions[t]
            
            # Buy new
            for t in new_holdings:
                if t not in positions and t in data and d in data[t].index:
                    p = data[t].loc[d, 'close']
                    positions[t] = {'entry': p, 'peak': p}
        
        # Daily return
        ret = 0.0
        active = [t for t in positions if t in data and d in data[t].index and d_next in data[t].index]
        if active:
            for t in active:
                r = data[t].loc[d_next, 'close'] / data[t].loc[d, 'close'] - 1
                ret += r / len(active)
        
        ret *= leverage
        daily_returns.append(ret)
        portfolio *= (1 + ret)
        
        if portfolio > peak_portfolio:
            peak_portfolio = portfolio
        dd = (portfolio - peak_portfolio) / peak_portfolio
        if dd < max_dd:
            max_dd = dd
    
    total_ret = portfolio / 100 - 1
    daily_returns = np.array(daily_returns)
    sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    return {
        'return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trailing_stop': trailing_stop
    }


def optimize_single_strategy(name: str, n_stocks: int, rebalance: int, leverage: bool):
    """Optimize exit parameters for one strategy."""
    
    print(f"\n{'='*80}")
    print(f"🔬 TEAM: {name}")
    print(f"{'='*80}")
    
    # Baseline
    baseline = run_strategy(n_stocks, rebalance, leverage, None, None, None)
    print(f"\n📊 BASELINE: Return={baseline['return']*100:+.1f}%, Sharpe={baseline['sharpe']:.2f}, MaxDD={baseline['max_dd']*100:.1f}%")
    
    # Parameter grid (focused)
    stop_losses = [None, 0.05, 0.08, 0.10, 0.15, 0.20]
    take_profits = [None, 0.15, 0.25, 0.40, 0.60, 1.00]
    trailing_stops = [None, 0.08, 0.12, 0.15, 0.20]
    
    results = []
    total = len(stop_losses) * len(take_profits) * len(trailing_stops)
    count = 0
    
    print(f"\n🔄 Testing {total} combinations...")
    
    for sl in stop_losses:
        for tp in take_profits:
            for ts in trailing_stops:
                count += 1
                if count % 30 == 0:
                    print(f"   {count}/{total}...")
                
                r = run_strategy(n_stocks, rebalance, leverage, sl, tp, ts)
                r['vs_baseline'] = r['return'] - baseline['return']
                results.append(r)
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    return results, baseline


def main():
    print("=" * 100)
    print("🏆 EXIT STRATEGY OPTIMIZATION - 4 TEAMS")
    print("=" * 100)
    
    load_all_data()
    
    strategies = [
        {'name': 'TEAM 1: Concentrated (Top 5)', 'n': 5, 'rebal': 21, 'lev': False},
        {'name': 'TEAM 2: Momentum + 1.5x', 'n': 15, 'rebal': 5, 'lev': True},
        {'name': 'TEAM 3: Quality Momentum', 'n': 20, 'rebal': 5, 'lev': False},
        {'name': 'TEAM 4: Adaptive', 'n': 12, 'rebal': 5, 'lev': False},
    ]
    
    all_results = {}
    
    for s in strategies:
        results, baseline = optimize_single_strategy(s['name'], s['n'], s['rebal'], s['lev'])
        all_results[s['name']] = {'results': results, 'baseline': baseline}
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 100)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 100)
    
    for name, data in all_results.items():
        results = data['results']
        baseline = data['baseline']
        
        print(f"\n{'─'*100}")
        print(f"📈 {name}")
        print(f"{'─'*100}")
        print(f"  BASELINE: {baseline['return']*100:+.1f}% return, {baseline['sharpe']:.2f} Sharpe, {baseline['max_dd']*100:.1f}% MaxDD")
        
        print(f"\n  🏆 TOP 10 BY TOTAL RETURN:")
        print(f"  {'#':<4} {'SL':<10} {'TP':<10} {'Trail':<10} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10} {'vs Base':<12}")
        print(f"  {'-'*85}")
        
        for i, r in enumerate(results[:10]):
            sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
            tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
            ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            print(f"  {rank:<4} {sl:<10} {tp:<10} {ts:<10} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}% {r['vs_baseline']*100:>+10.1f}%")
        
        # Best by Sharpe
        by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        print(f"\n  📊 TOP 5 BY SHARPE RATIO:")
        print(f"  {'#':<4} {'SL':<10} {'TP':<10} {'Trail':<10} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10}")
        print(f"  {'-'*75}")
        
        for i, r in enumerate(by_sharpe[:5]):
            sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
            tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
            ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            print(f"  {rank:<4} {sl:<10} {tp:<10} {ts:<10} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}%")
        
        # Best by lowest DD (profitable only)
        profitable = [r for r in results if r['return'] > 0]
        if profitable:
            by_dd = sorted(profitable, key=lambda x: x['max_dd'], reverse=True)
            print(f"\n  🛡️ TOP 5 BY LOWEST DRAWDOWN (Profitable):")
            print(f"  {'#':<4} {'SL':<10} {'TP':<10} {'Trail':<10} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10}")
            print(f"  {'-'*75}")
            
            for i, r in enumerate(by_dd[:5]):
                sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
                tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
                ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                print(f"  {rank:<4} {sl:<10} {tp:<10} {ts:<10} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}%")
    
    # =========================================================================
    # WINNER TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("🏆 OPTIMAL PARAMETERS PER STRATEGY")
    print("=" * 100)
    
    print(f"\n  {'Strategy':<35} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Δ vs Base':<12}")
    print(f"  {'-'*100}")
    
    for name, data in all_results.items():
        best = data['results'][0]
        baseline = data['baseline']
        
        sl = f"{best['stop_loss']*100:.0f}%" if best['stop_loss'] else "None"
        tp = f"{best['take_profit']*100:.0f}%" if best['take_profit'] else "None"
        ts = f"{best['trailing_stop']*100:.0f}%" if best['trailing_stop'] else "None"
        improvement = best['return'] - baseline['return']
        
        emoji = "🚀" if improvement > 0.10 else "✅" if improvement > 0 else "➖" if improvement > -0.05 else "⚠️"
        short_name = name.replace("TEAM ", "").replace(": ", " - ")
        print(f"  {emoji} {short_name:<33} {sl:<12} {tp:<14} {ts:<12} {best['return']*100:>+10.1f}% {improvement*100:>+10.1f}%")
    
    # =========================================================================
    # INSIGHTS
    # =========================================================================
    print("\n" + "=" * 100)
    print("💡 KEY INSIGHTS")
    print("=" * 100)
    
    for name, data in all_results.items():
        results = data['results']
        baseline = data['baseline']
        best = results[0]
        
        improved = sum(1 for r in results if r['return'] > baseline['return'])
        total = len(results)
        
        short_name = name.replace("TEAM ", "").replace(": ", " - ")
        
        print(f"\n  {short_name}:")
        print(f"  ───────────────────────────────────────────────")
        print(f"  • {improved}/{total} configs beat baseline ({improved/total*100:.0f}%)")
        print(f"  • Best return: {best['return']*100:+.1f}% vs baseline {baseline['return']*100:+.1f}%")
        
        if best['stop_loss'] is None and best['take_profit'] is None and best['trailing_stop'] is None:
            print(f"  • 📌 FINDING: NO EXITS is optimal - momentum works best uninterrupted")
        else:
            if best['stop_loss']:
                print(f"  • Stop-Loss: {best['stop_loss']*100:.0f}% helps limit downside")
            if best['take_profit']:
                print(f"  • Take-Profit: {best['take_profit']*100:.0f}% locks in gains")
            if best['trailing_stop']:
                print(f"  • Trailing: {best['trailing_stop']*100:.0f}% protects profits")
    
    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 100)
    
    for name, data in all_results.items():
        best = data['results'][0]
        baseline = data['baseline']
        improvement = best['return'] - baseline['return']
        
        sl = f"{best['stop_loss']*100:.0f}%" if best['stop_loss'] else "None"
        tp = f"{best['take_profit']*100:.0f}%" if best['take_profit'] else "None"
        ts = f"{best['trailing_stop']*100:.0f}%" if best['trailing_stop'] else "None"
        
        short_name = name.replace("TEAM ", "").replace(": ", " - ")
        
        if improvement > 0.05:
            verdict = "USE OPTIMIZED EXITS"
        elif improvement > -0.02:
            verdict = "EXITS OPTIONAL"
        else:
            verdict = "KEEP ORIGINAL (NO EXITS)"
        
        print(f"""
  📌 {short_name}:
     Recommendation: {verdict}
     Optimal Config: SL={sl} | TP={tp} | Trailing={ts}
     Expected Improvement: {improvement*100:+.1f}%
        """)
    
    print("=" * 100)


if __name__ == "__main__":
    main()
