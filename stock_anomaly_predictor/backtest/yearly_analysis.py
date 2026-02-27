#!/usr/bin/env python3
"""
YEARLY PERFORMANCE ANALYSIS
===========================
Shows year-by-year performance for each strategy.
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


# High-quality TASI stocks
TASI_TICKERS = [
    '1180.SR', '1010.SR', '1150.SR', '1140.SR',
    '2222.SR', '2082.SR', '2010.SR', '2020.SR', '2350.SR',
    '1211.SR', '1304.SR', '4190.SR', '4001.SR', '4007.SR',
    '7010.SR', '7020.SR', '8010.SR', '8210.SR', '4300.SR', '4280.SR',
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
    vol_penalty = vol * 2
    
    mom = m1 * 0.3 + m3 * 0.3 + m6 * 0.25 + m12 * 0.15
    
    trend_bonus = 0
    if c.iloc[idx] > ma50:
        trend_bonus += 0.05
    if ma50 > ma200:
        trend_bonus += 0.05
    
    return mom + trend_bonus - vol_penalty


def run_strategies(data, dates, start_idx, market):
    """Run all strategies and return daily equity curves with dates."""
    
    strategies = {}
    
    # =========================================================================
    # STRATEGY 1: Buy & Hold (Baseline)
    # =========================================================================
    portfolio = [100]
    port_dates = [dates[start_idx]]
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
        port_dates.append(d_next)
    
    strategies['Buy & Hold'] = {'equity': portfolio, 'dates': port_dates}
    
    # =========================================================================
    # STRATEGY 2: Quality Momentum (Weekly)
    # =========================================================================
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        if (i - start_idx) % 5 == 0 or i == start_idx:
            scores = []
            for t, df in data.items():
                if d in df.index:
                    idx = df.index.get_loc(d)
                    s = quality_momentum_score(df, idx)
                    if s > -900:
                        scores.append((t, s))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            n_hold = max(5, len([s for s in scores if s[1] > 0]) // 2)
            holdings = [s[0] for s in scores[:n_hold]]
        
        ret = 0
        for t in holdings:
            df = data[t]
            if d in df.index and d_next in df.index:
                r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                ret += r / len(holdings)
        
        portfolio.append(portfolio[-1] * (1 + ret))
        port_dates.append(d_next)
    
    strategies['Quality Momentum'] = {'equity': portfolio, 'dates': port_dates}
    
    # =========================================================================
    # STRATEGY 3: Momentum + 1.5x Leverage
    # =========================================================================
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Market regime
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
        port_dates.append(d_next)
    
    strategies['Momentum + 1.5x Leverage'] = {'equity': portfolio, 'dates': port_dates}
    
    # =========================================================================
    # STRATEGY 4: Concentrated (Top 5)
    # =========================================================================
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
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
        port_dates.append(d_next)
    
    strategies['Concentrated (Top 5)'] = {'equity': portfolio, 'dates': port_dates}
    
    # =========================================================================
    # STRATEGY 5: Adaptive
    # =========================================================================
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        if d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 200:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                ma200 = market['close'].iloc[idx-200:idx].mean()
                
                if p > ma50 > ma200:
                    n_hold = 10
                    leverage = 1.3
                elif p > ma50:
                    n_hold = 7
                    leverage = 1.0
                elif p > ma200:
                    n_hold = 5
                    leverage = 0.8
                else:
                    n_hold = 3
                    leverage = 0.5
            else:
                n_hold = 7
                leverage = 1.0
        else:
            n_hold = 7
            leverage = 1.0
        
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
        port_dates.append(d_next)
    
    strategies['Adaptive'] = {'equity': portfolio, 'dates': port_dates}
    
    return strategies


def calculate_yearly_returns(strategies):
    """Calculate year-by-year returns for each strategy."""
    
    yearly_data = {}
    
    for name, data in strategies.items():
        equity = data['equity']
        dates = data['dates']
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'equity': equity
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        
        # Calculate yearly returns
        yearly_returns = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            if len(year_data) > 1:
                start_val = year_data['equity'].iloc[0]
                end_val = year_data['equity'].iloc[-1]
                ret = (end_val / start_val - 1) * 100
                yearly_returns[year] = ret
        
        yearly_data[name] = yearly_returns
    
    return yearly_data


def run():
    print("=" * 120)
    print("📊 YEARLY PERFORMANCE ANALYSIS")
    print("=" * 120)
    
    data = fetch_data('2019-01-01')
    
    all_dates = set.intersection(*[set(df.index) for df in data.values()])
    dates = sorted(list(all_dates))
    start_idx = 252
    
    market = data.get('2222.SR', list(data.values())[0])
    
    print("\nRunning all strategies...")
    strategies = run_strategies(data, dates, start_idx, market)
    
    print("Calculating yearly returns...\n")
    yearly = calculate_yearly_returns(strategies)
    
    # Get all years
    all_years = set()
    for name, years in yearly.items():
        all_years.update(years.keys())
    all_years = sorted(all_years)
    
    # Print yearly comparison table
    print("=" * 120)
    print("📅 YEAR-BY-YEAR PERFORMANCE COMPARISON")
    print("=" * 120)
    
    # Header
    header = f"  {'Strategy':<30}"
    for year in all_years:
        header += f" {year:>10}"
    header += f" {'TOTAL':>12}"
    print(header)
    print("  " + "-" * (30 + len(all_years) * 11 + 13))
    
    # Strategy order (by total return)
    strategy_order = [
        'Momentum + 1.5x Leverage',
        'Concentrated (Top 5)',
        'Quality Momentum',
        'Adaptive',
        'Buy & Hold'
    ]
    
    for name in strategy_order:
        if name not in yearly:
            continue
        
        years_data = yearly[name]
        
        # Calculate total
        total = 1
        for y in all_years:
            if y in years_data:
                total *= (1 + years_data[y] / 100)
        total = (total - 1) * 100
        
        # Determine emoji
        if total > 80:
            emoji = "🏆"
        elif total > 50:
            emoji = "🥇"
        elif total > 30:
            emoji = "🥈"
        elif total > 0:
            emoji = "✅"
        else:
            emoji = "❌"
        
        row = f"  {emoji} {name:<28}"
        for year in all_years:
            if year in years_data:
                ret = years_data[year]
                if ret > 20:
                    row += f" {ret:>+9.1f}%"
                elif ret > 0:
                    row += f" {ret:>+9.1f}%"
                elif ret > -10:
                    row += f" {ret:>+9.1f}%"
                else:
                    row += f" {ret:>+9.1f}%"
            else:
                row += f" {'N/A':>10}"
        
        row += f" {total:>+11.1f}%"
        print(row)
    
    # Best/Worst year analysis
    print(f"\n{'='*120}")
    print("📈 BEST & WORST YEARS BY STRATEGY")
    print("=" * 120)
    
    for name in strategy_order:
        if name not in yearly:
            continue
        
        years_data = yearly[name]
        if not years_data:
            continue
        
        best_year = max(years_data.items(), key=lambda x: x[1])
        worst_year = min(years_data.items(), key=lambda x: x[1])
        
        pos_years = sum(1 for v in years_data.values() if v > 0)
        neg_years = sum(1 for v in years_data.values() if v < 0)
        
        print(f"\n  {name}:")
        print(f"    Best Year:   {best_year[0]} ({best_year[1]:+.1f}%)")
        print(f"    Worst Year:  {worst_year[0]} ({worst_year[1]:+.1f}%)")
        print(f"    Positive Years: {pos_years}/{len(years_data)}")
    
    # Detailed yearly table
    print(f"\n{'='*120}")
    print("📋 DETAILED YEARLY BREAKDOWN")
    print("=" * 120)
    
    for year in all_years:
        print(f"\n  {year}:")
        print(f"  {'-'*60}")
        
        year_results = []
        for name in strategy_order:
            if name in yearly and year in yearly[name]:
                year_results.append((name, yearly[name][year]))
        
        year_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, ret) in enumerate(year_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            color_indicator = "📈" if ret > 0 else "📉"
            print(f"    {rank} {name:<30} {color_indicator} {ret:>+8.1f}%")
    
    # Summary statistics
    print(f"\n{'='*120}")
    print("📊 SUMMARY STATISTICS")
    print("=" * 120)
    
    print(f"\n  {'Strategy':<30} {'Avg Year':>10} {'Best':>10} {'Worst':>10} {'Win%':>10} {'Volatility':>12}")
    print("  " + "-" * 85)
    
    for name in strategy_order:
        if name not in yearly:
            continue
        
        years_data = yearly[name]
        returns = list(years_data.values())
        
        avg = np.mean(returns)
        best = max(returns)
        worst = min(returns)
        win_pct = sum(1 for r in returns if r > 0) / len(returns) * 100
        vol = np.std(returns)
        
        print(f"  {name:<30} {avg:>+9.1f}% {best:>+9.1f}% {worst:>+9.1f}% {win_pct:>9.0f}% {vol:>11.1f}%")
    
    print(f"\n{'='*120}")
    print("END OF YEARLY ANALYSIS")
    print("=" * 120)


if __name__ == "__main__":
    run()
