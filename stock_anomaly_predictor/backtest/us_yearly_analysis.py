#!/usr/bin/env python3
"""
US MARKET YEARLY PERFORMANCE ANALYSIS
======================================
Top 200 US stocks with same strategies as TASI analysis.
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


# Top 200 US Stocks (by market cap - mix of S&P 500 leaders)
US_TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'CSCO', 'ACN', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW',
    'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'PANW',
    'CRWD', 'FTNT', 'WDAY', 'TEAM', 'ZS', 'DDOG', 'NET', 'SNOW', 'PLTR', 'UBER',
    
    # Finance
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C',
    'BLK', 'SCHW', 'SPGI', 'CME', 'ICE', 'PGR', 'AON', 'MMC', 'CB', 'MET',
    'AIG', 'TRV', 'AFL', 'PRU', 'ALL', 'COF', 'USB', 'PNC', 'TFC', 'BK',
    
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    'AMGN', 'GILD', 'MDT', 'ISRG', 'CVS', 'ELV', 'CI', 'SYK', 'BSX', 'VRTX',
    'REGN', 'ZTS', 'BDX', 'HUM', 'MCK', 'CAH', 'DXCM', 'IDXX', 'IQV', 'A',
    
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    'LOW', 'TJX', 'EL', 'CL', 'MDLZ', 'KMB', 'GIS', 'K', 'HSY', 'SYY',
    'DG', 'DLTR', 'ROST', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU', 'CMG', 'YUM',
    'DPZ', 'SBAC', 'POOL', 'TSCO', 'DRI', 'LVS', 'WYNN', 'MGM', 'MAR', 'HLT',
    
    # Industrial
    'CAT', 'DE', 'UNP', 'UPS', 'HON', 'RTX', 'BA', 'LMT', 'GE', 'MMM',
    'GD', 'NOC', 'FDX', 'CSX', 'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW',
    'PH', 'ROK', 'CMI', 'PCAR', 'FAST', 'ODFL', 'JBHT', 'CHRW', 'EXPD', 'XPO',
    
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'PXD',
    'DVN', 'HES', 'HAL', 'BKR', 'FANG', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    
    # Communications
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'WBD',
    
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'WELL', 'SPG', 'AVB',
    
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC',
]


def fetch_data(start: str = '2019-01-01'):
    print(f"Fetching data for {len(US_TICKERS)} US stocks from {start}...")
    data = {}
    failed = 0
    
    for i, t in enumerate(US_TICKERS):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(US_TICKERS)} stocks loaded...")
        try:
            df = yf.Ticker(t).history(start=start)
            if len(df) >= 252:
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)
                data[t] = df
        except:
            failed += 1
    
    print(f"Successfully loaded {len(data)} stocks (failed: {failed})")
    return data


def fetch_market_index(start: str = '2019-01-01'):
    """Fetch S&P 500 as market proxy."""
    print("Fetching S&P 500 index...")
    try:
        df = yf.Ticker('SPY').history(start=start)
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_localize(None)
        return df
    except:
        return None


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
    
    # Quality filter: must be above 95% of 200MA
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
    print("  Running Buy & Hold...")
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
    print("  Running Quality Momentum...")
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
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
            n_hold = max(10, len([s for s in scores if s[1] > 0]) // 4)
            n_hold = min(n_hold, 50)  # Cap at 50 stocks
            holdings = [s[0] for s in scores[:n_hold]]
        
        ret = 0
        if holdings:
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
    print("  Running Momentum + 1.5x Leverage...")
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Market regime check using SPY
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
            # Top 20 stocks for US market
            holdings = [s[0] for s in scores[:20] if s[1] > 0]
            if len(holdings) < 10:
                holdings = [s[0] for s in scores[:10]]
        
        ret = 0
        if holdings:
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
    print("  Running Concentrated (Top 5)...")
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Rebalance monthly
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
        if holdings:
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
    print("  Running Adaptive...")
    portfolio = [100]
    port_dates = [dates[start_idx]]
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
        # Dynamic regime detection
        if d in market.index:
            idx = market.index.get_loc(d)
            if idx >= 200:
                p = market['close'].iloc[idx]
                ma50 = market['close'].iloc[idx-50:idx].mean()
                ma200 = market['close'].iloc[idx-200:idx].mean()
                
                if p > ma50 > ma200:  # Strong bull
                    n_hold = 25
                    leverage = 1.3
                elif p > ma50:  # Moderate bull
                    n_hold = 20
                    leverage = 1.0
                elif p > ma200:  # Weak/sideways
                    n_hold = 15
                    leverage = 0.8
                else:  # Bear
                    n_hold = 10
                    leverage = 0.5
            else:
                n_hold = 15
                leverage = 1.0
        else:
            n_hold = 15
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
        if holdings:
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
    
    for name, strat_data in strategies.items():
        equity = strat_data['equity']
        dates = strat_data['dates']
        
        df = pd.DataFrame({
            'date': dates,
            'equity': equity
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        
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
    print("📊 US MARKET YEARLY PERFORMANCE ANALYSIS (Top 200 Stocks)")
    print("=" * 120)
    
    data = fetch_data('2019-01-01')
    market = fetch_market_index('2019-01-01')
    
    if market is None:
        print("Failed to fetch market index, using first stock as proxy")
        market = list(data.values())[0]
    
    # Find common dates
    all_dates = set(market.index)
    for df in data.values():
        all_dates &= set(df.index)
    
    dates = sorted(list(all_dates))
    start_idx = 252  # Start after 1 year of data
    
    print(f"\nDate range: {dates[start_idx].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(dates) - start_idx}")
    
    print("\nRunning all strategies...")
    strategies = run_strategies(data, dates, start_idx, market)
    
    print("\nCalculating yearly returns...")
    yearly = calculate_yearly_returns(strategies)
    
    # Get all years
    all_years = set()
    for name, years in yearly.items():
        all_years.update(years.keys())
    all_years = sorted(all_years)
    
    # Print yearly comparison
    print("\n" + "=" * 120)
    print("📅 YEAR-BY-YEAR PERFORMANCE COMPARISON (US MARKET)")
    print("=" * 120)
    
    header = f"  {'Strategy':<30}"
    for year in all_years:
        header += f" {year:>10}"
    header += f" {'TOTAL':>12}"
    print(header)
    print("  " + "-" * (30 + len(all_years) * 11 + 13))
    
    strategy_order = [
        'Momentum + 1.5x Leverage',
        'Concentrated (Top 5)',
        'Quality Momentum',
        'Adaptive',
        'Buy & Hold'
    ]
    
    totals = {}
    for name in strategy_order:
        if name not in yearly:
            continue
        
        years_data = yearly[name]
        
        total = 1
        for y in all_years:
            if y in years_data:
                total *= (1 + years_data[y] / 100)
        total = (total - 1) * 100
        totals[name] = total
        
        if total > 150:
            emoji = "🏆"
        elif total > 100:
            emoji = "🥇"
        elif total > 50:
            emoji = "🥈"
        elif total > 0:
            emoji = "✅"
        else:
            emoji = "❌"
        
        row = f"  {emoji} {name:<28}"
        for year in all_years:
            if year in years_data:
                ret = years_data[year]
                row += f" {ret:>+9.1f}%"
            else:
                row += f" {'N/A':>10}"
        
        row += f" {total:>+11.1f}%"
        print(row)
    
    # Best/Worst analysis
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
    
    # Detailed breakdown
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
            indicator = "📈" if ret > 0 else "📉"
            print(f"    {rank} {name:<30} {indicator} {ret:>+8.1f}%")
    
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
    print("END OF US MARKET ANALYSIS")
    print("=" * 120)


if __name__ == "__main__":
    run()
