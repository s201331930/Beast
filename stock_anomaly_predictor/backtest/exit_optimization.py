#!/usr/bin/env python3
"""
EXIT STRATEGY OPTIMIZATION
==========================
4 Teams optimizing stop-loss, take-profit, and trailing stops for each strategy.

Team 1: Concentrated (Top 5)
Team 2: Momentum + 1.5x Leverage  
Team 3: Quality Momentum
Team 4: Adaptive
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


# Top US Stocks (reduced for faster optimization)
US_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT',
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK',
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'AMGN',
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW',
    'CAT', 'DE', 'UNP', 'HON', 'RTX', 'BA', 'LMT', 'GE', 'EMR', 'ETN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS',
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX',
    'NEE', 'DUK', 'SO', 'D', 'AEP',
]


@dataclass
class TradeResult:
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    return_pct: float
    exit_reason: str
    holding_days: int


def fetch_data(start: str = '2019-01-01'):
    print(f"Fetching data for {len(US_TICKERS)} stocks...")
    data = {}
    for i, t in enumerate(US_TICKERS):
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


def fetch_market(start: str = '2019-01-01'):
    df = yf.Ticker('SPY').history(start=start)
    df.columns = [c.lower() for c in df.columns]
    df.index = df.index.tz_localize(None)
    return df


def quality_momentum_score(df, idx):
    if idx < 252:
        return -999
    c = df['close']
    m1 = c.iloc[idx] / c.iloc[idx-21] - 1
    m3 = c.iloc[idx] / c.iloc[idx-63] - 1
    m6 = c.iloc[idx] / c.iloc[idx-126] - 1
    m12 = c.iloc[idx] / c.iloc[idx-252] - 1
    
    ma50 = c.iloc[idx-50:idx].mean()
    ma200 = c.iloc[idx-200:idx].mean()
    
    if c.iloc[idx] < ma200 * 0.95:
        return -999
    
    vol = c.pct_change().iloc[idx-60:idx].std()
    mom = m1 * 0.3 + m3 * 0.3 + m6 * 0.25 + m12 * 0.15
    trend_bonus = 0.05 if c.iloc[idx] > ma50 else 0
    trend_bonus += 0.05 if ma50 > ma200 else 0
    
    return mom + trend_bonus - vol * 2


def simulate_with_exits(data: Dict, dates: List, start_idx: int, market: pd.DataFrame,
                        strategy: str, n_stocks: int, rebalance_freq: int,
                        stop_loss: Optional[float], take_profit: Optional[float],
                        trailing_stop: Optional[float], use_leverage: bool = False) -> Tuple[float, float, float]:
    """
    Simulate strategy with exit conditions.
    
    Returns: (total_return, sharpe, max_drawdown)
    """
    portfolio = 100.0
    peak = 100.0
    max_dd = 0.0
    daily_returns = []
    
    # Track positions: {ticker: {'entry_price': x, 'peak_price': x, 'entry_idx': i}}
    positions = {}
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Determine leverage (for Momentum 1.5x strategy)
        leverage = 1.0
        if use_leverage and d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 50:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                leverage = 1.5 if p > ma50 else 1.0
        
        # Check for exits on current positions
        stocks_to_sell = []
        for ticker in list(positions.keys()):
            if ticker not in data or d not in data[ticker].index:
                stocks_to_sell.append(ticker)
                continue
            
            df = data[ticker]
            current_price = df.loc[d, 'close']
            pos = positions[ticker]
            entry_price = pos['entry_price']
            peak_price = pos['peak_price']
            
            # Update peak price for trailing stop
            if current_price > peak_price:
                positions[ticker]['peak_price'] = current_price
                peak_price = current_price
            
            # Check stop-loss
            if stop_loss is not None:
                loss = (current_price - entry_price) / entry_price
                if loss <= -stop_loss:
                    stocks_to_sell.append(ticker)
                    continue
            
            # Check take-profit
            if take_profit is not None:
                gain = (current_price - entry_price) / entry_price
                if gain >= take_profit:
                    stocks_to_sell.append(ticker)
                    continue
            
            # Check trailing stop
            if trailing_stop is not None:
                drawdown_from_peak = (current_price - peak_price) / peak_price
                if drawdown_from_peak <= -trailing_stop:
                    stocks_to_sell.append(ticker)
                    continue
        
        # Remove sold positions
        for ticker in stocks_to_sell:
            if ticker in positions:
                del positions[ticker]
        
        # Rebalancing logic
        should_rebalance = (i - start_idx) % rebalance_freq == 0 or i == start_idx
        
        if should_rebalance:
            # Score all stocks
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            new_holdings = [s[0] for s in scores[:n_stocks]]
            
            # Sell positions not in new holdings
            for ticker in list(positions.keys()):
                if ticker not in new_holdings:
                    del positions[ticker]
            
            # Add new positions
            for ticker in new_holdings:
                if ticker not in positions and ticker in data and d in data[ticker].index:
                    entry_price = data[ticker].loc[d, 'close']
                    positions[ticker] = {
                        'entry_price': entry_price,
                        'peak_price': entry_price,
                        'entry_idx': i
                    }
            
            holdings = new_holdings
        
        # Calculate daily return
        daily_ret = 0.0
        active_positions = [t for t in positions.keys() if t in data and d in data[t].index and d_next in data[t].index]
        
        if active_positions:
            for ticker in active_positions:
                df = data[ticker]
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                daily_ret += r / len(active_positions)
        
        daily_ret *= leverage
        daily_returns.append(daily_ret)
        
        portfolio *= (1 + daily_ret)
        if portfolio > peak:
            peak = portfolio
        dd = (portfolio - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    # Calculate metrics
    total_return = (portfolio / 100) - 1
    
    if len(daily_returns) > 0:
        daily_returns = np.array(daily_returns)
        avg_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        sharpe = (avg_daily * 252) / (std_daily * np.sqrt(252)) if std_daily > 0 else 0
    else:
        sharpe = 0
    
    return total_return, sharpe, max_dd


def optimize_strategy(data, dates, start_idx, market, strategy_name, n_stocks, rebalance_freq, use_leverage):
    """Run optimization for a single strategy."""
    
    print(f"\n{'='*80}")
    print(f"🔬 TEAM OPTIMIZING: {strategy_name}")
    print(f"{'='*80}")
    
    # Parameter grid
    stop_losses = [None, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    take_profits = [None, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
    trailing_stops = [None, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    # First: Test baseline (no exits)
    baseline_ret, baseline_sharpe, baseline_dd = simulate_with_exits(
        data, dates, start_idx, market, strategy_name, n_stocks, rebalance_freq,
        None, None, None, use_leverage
    )
    
    print(f"\n📊 BASELINE (No Exits):")
    print(f"   Return: {baseline_ret*100:+.1f}%  |  Sharpe: {baseline_sharpe:.2f}  |  Max DD: {baseline_dd*100:.1f}%")
    
    results = []
    total_combos = len(stop_losses) * len(take_profits) * len(trailing_stops)
    combo_num = 0
    
    print(f"\n🔄 Testing {total_combos} parameter combinations...")
    
    for sl in stop_losses:
        for tp in take_profits:
            for ts in trailing_stops:
                combo_num += 1
                if combo_num % 50 == 0:
                    print(f"   Progress: {combo_num}/{total_combos}")
                
                ret, sharpe, dd = simulate_with_exits(
                    data, dates, start_idx, market, strategy_name, n_stocks, rebalance_freq,
                    sl, tp, ts, use_leverage
                )
                
                results.append({
                    'stop_loss': sl,
                    'take_profit': tp,
                    'trailing_stop': ts,
                    'return': ret,
                    'sharpe': sharpe,
                    'max_dd': dd,
                    'return_vs_baseline': ret - baseline_ret
                })
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    return results, baseline_ret, baseline_sharpe, baseline_dd


def run():
    print("=" * 100)
    print("🏆 EXIT STRATEGY OPTIMIZATION - 4 TEAMS CHALLENGE")
    print("=" * 100)
    
    data = fetch_data('2019-01-01')
    market = fetch_market('2019-01-01')
    
    all_dates = set(market.index)
    for df in data.values():
        all_dates &= set(df.index)
    dates = sorted(list(all_dates))
    start_idx = 252
    
    print(f"\nDate range: {dates[start_idx].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Define strategies
    strategies = [
        {'name': 'Concentrated (Top 5)', 'n_stocks': 5, 'rebalance': 21, 'leverage': False},
        {'name': 'Momentum + 1.5x Leverage', 'n_stocks': 20, 'rebalance': 5, 'leverage': True},
        {'name': 'Quality Momentum', 'n_stocks': 25, 'rebalance': 5, 'leverage': False},
        {'name': 'Adaptive', 'n_stocks': 15, 'rebalance': 5, 'leverage': False},
    ]
    
    all_results = {}
    
    for strat in strategies:
        results, baseline_ret, baseline_sharpe, baseline_dd = optimize_strategy(
            data, dates, start_idx, market,
            strat['name'], strat['n_stocks'], strat['rebalance'], strat['leverage']
        )
        all_results[strat['name']] = {
            'results': results,
            'baseline': {'return': baseline_ret, 'sharpe': baseline_sharpe, 'max_dd': baseline_dd}
        }
    
    # =========================================================================
    # COMPREHENSIVE RESULTS
    # =========================================================================
    print("\n" + "=" * 100)
    print("📊 OPTIMIZATION RESULTS BY STRATEGY")
    print("=" * 100)
    
    for strat_name, data in all_results.items():
        results = data['results']
        baseline = data['baseline']
        
        print(f"\n{'─'*100}")
        print(f"📈 {strat_name}")
        print(f"{'─'*100}")
        
        print(f"\n  BASELINE (No Exits): Return={baseline['return']*100:+.1f}%, Sharpe={baseline['sharpe']:.2f}, MaxDD={baseline['max_dd']*100:.1f}%")
        
        # Top 5 by return
        print(f"\n  TOP 5 BY TOTAL RETURN:")
        print(f"  {'Rank':<6} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10} {'vs Base':<12}")
        print(f"  {'-'*90}")
        
        for i, r in enumerate(results[:5]):
            sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
            tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
            ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"  {emoji} #{i+1:<3} {sl:<12} {tp:<14} {ts:<12} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}% {r['return_vs_baseline']*100:>+10.1f}%")
        
        # Top 5 by Sharpe
        by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
        print(f"\n  TOP 5 BY SHARPE RATIO:")
        print(f"  {'Rank':<6} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10}")
        print(f"  {'-'*80}")
        
        for i, r in enumerate(by_sharpe[:5]):
            sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
            tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
            ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"  {emoji} #{i+1:<3} {sl:<12} {tp:<14} {ts:<12} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}%")
        
        # Top 5 by lowest drawdown (among profitable)
        profitable = [r for r in results if r['return'] > 0]
        if profitable:
            by_dd = sorted(profitable, key=lambda x: x['max_dd'], reverse=True)  # Less negative = better
            print(f"\n  TOP 5 BY LOWEST DRAWDOWN (Profitable Only):")
            print(f"  {'Rank':<6} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10}")
            print(f"  {'-'*80}")
            
            for i, r in enumerate(by_dd[:5]):
                sl = f"{r['stop_loss']*100:.0f}%" if r['stop_loss'] else "None"
                tp = f"{r['take_profit']*100:.0f}%" if r['take_profit'] else "None"
                ts = f"{r['trailing_stop']*100:.0f}%" if r['trailing_stop'] else "None"
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"  {emoji} #{i+1:<3} {sl:<12} {tp:<14} {ts:<12} {r['return']*100:>+10.1f}% {r['sharpe']:>9.2f} {r['max_dd']*100:>9.1f}%")
    
    # =========================================================================
    # WINNER SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("🏆 OPTIMAL EXIT PARAMETERS BY STRATEGY")
    print("=" * 100)
    
    print(f"\n  {'Strategy':<30} {'Stop-Loss':<12} {'Take-Profit':<14} {'Trailing':<12} {'Return':<12} {'Improvement':<12}")
    print(f"  {'-'*95}")
    
    for strat_name, data in all_results.items():
        best = data['results'][0]
        baseline = data['baseline']
        
        sl = f"{best['stop_loss']*100:.0f}%" if best['stop_loss'] else "None"
        tp = f"{best['take_profit']*100:.0f}%" if best['take_profit'] else "None"
        ts = f"{best['trailing_stop']*100:.0f}%" if best['trailing_stop'] else "None"
        improvement = best['return'] - baseline['return']
        
        emoji = "✅" if improvement > 0 else "❌"
        print(f"  {emoji} {strat_name:<28} {sl:<12} {tp:<14} {ts:<12} {best['return']*100:>+10.1f}% {improvement*100:>+10.1f}%")
    
    # =========================================================================
    # DETAILED ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("📋 DETAILED ANALYSIS & INSIGHTS")
    print("=" * 100)
    
    for strat_name, data in all_results.items():
        results = data['results']
        baseline = data['baseline']
        
        # Find best configs
        best_return = results[0]
        best_sharpe = max(results, key=lambda x: x['sharpe'])
        
        # Analyze what works
        improved = [r for r in results if r['return'] > baseline['return']]
        worse = [r for r in results if r['return'] < baseline['return']]
        
        print(f"\n  {strat_name}:")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  • Configurations tested: {len(results)}")
        print(f"  • Improved vs baseline: {len(improved)} ({len(improved)/len(results)*100:.0f}%)")
        print(f"  • Worse vs baseline: {len(worse)} ({len(worse)/len(results)*100:.0f}%)")
        
        print(f"\n  📈 BEST FOR RETURNS:")
        sl = f"{best_return['stop_loss']*100:.0f}%" if best_return['stop_loss'] else "None"
        tp = f"{best_return['take_profit']*100:.0f}%" if best_return['take_profit'] else "None"
        ts = f"{best_return['trailing_stop']*100:.0f}%" if best_return['trailing_stop'] else "None"
        print(f"     Stop-Loss: {sl}, Take-Profit: {tp}, Trailing: {ts}")
        print(f"     Return: {best_return['return']*100:+.1f}% (vs {baseline['return']*100:+.1f}% baseline)")
        print(f"     Improvement: {(best_return['return']-baseline['return'])*100:+.1f}%")
        
        print(f"\n  📊 BEST FOR RISK-ADJUSTED:")
        sl = f"{best_sharpe['stop_loss']*100:.0f}%" if best_sharpe['stop_loss'] else "None"
        tp = f"{best_sharpe['take_profit']*100:.0f}%" if best_sharpe['take_profit'] else "None"
        ts = f"{best_sharpe['trailing_stop']*100:.0f}%" if best_sharpe['trailing_stop'] else "None"
        print(f"     Stop-Loss: {sl}, Take-Profit: {tp}, Trailing: {ts}")
        print(f"     Sharpe: {best_sharpe['sharpe']:.2f} (vs {baseline['sharpe']:.2f} baseline)")
    
    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 100)
    
    print("""
    Based on exhaustive optimization across 336+ parameter combinations per strategy:
    """)
    
    for strat_name, data in all_results.items():
        best = data['results'][0]
        baseline = data['baseline']
        improvement = best['return'] - baseline['return']
        
        sl = f"{best['stop_loss']*100:.0f}%" if best['stop_loss'] else "None"
        tp = f"{best['take_profit']*100:.0f}%" if best['take_profit'] else "None"
        ts = f"{best['trailing_stop']*100:.0f}%" if best['trailing_stop'] else "None"
        
        if improvement > 0.05:
            verdict = "SIGNIFICANT IMPROVEMENT"
            emoji = "🚀"
        elif improvement > 0:
            verdict = "MARGINAL IMPROVEMENT"
            emoji = "✅"
        elif improvement > -0.05:
            verdict = "NO SIGNIFICANT CHANGE"
            emoji = "➖"
        else:
            verdict = "EXITS HURT PERFORMANCE"
            emoji = "⚠️"
        
        print(f"""
    {emoji} {strat_name}:
       Optimal Config: SL={sl}, TP={tp}, Trailing={ts}
       Result: {best['return']*100:+.1f}% return ({improvement*100:+.1f}% vs baseline)
       Verdict: {verdict}
        """)
    
    print("=" * 100)
    print("END OF OPTIMIZATION")
    print("=" * 100)


if __name__ == "__main__":
    run()
