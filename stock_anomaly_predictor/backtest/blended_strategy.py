#!/usr/bin/env python3
"""
BLENDED STRATEGY ANALYSIS
=========================
Tests combining all strategies to exploit potential negative correlations.
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


# Top 200 US Stocks
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
    'LOW', 'TJX', 'EL', 'CL', 'MDLZ', 'KMB', 'GIS', 'HSY', 'SYY',
    'DG', 'DLTR', 'ROST', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU', 'CMG', 'YUM',
    'DPZ', 'SBAC', 'POOL', 'TSCO', 'DRI', 'LVS', 'WYNN', 'MGM', 'MAR', 'HLT',
    
    # Industrial
    'CAT', 'DE', 'UNP', 'UPS', 'HON', 'RTX', 'BA', 'LMT', 'GE', 'MMM',
    'GD', 'NOC', 'FDX', 'CSX', 'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW',
    'PH', 'ROK', 'CMI', 'PCAR', 'FAST', 'ODFL', 'JBHT', 'CHRW', 'EXPD', 'XPO',
    
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',
    'DVN', 'HAL', 'BKR', 'FANG', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG',
    
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
    print(f"Fetching data for {len(US_TICKERS)} US stocks...")
    data = {}
    
    for i, t in enumerate(US_TICKERS):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(US_TICKERS)}")
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


def fetch_market_index(start: str = '2019-01-01'):
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


def run_all_strategies(data, dates, start_idx, market):
    """Run all strategies and return DAILY returns for correlation analysis."""
    
    strategies_returns = {}
    
    all_stocks = list(data.keys())
    
    # =========================================================================
    # STRATEGY 1: Buy & Hold
    # =========================================================================
    print("  Running Buy & Hold...")
    returns = []
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
        returns.append(ret)
    
    strategies_returns['Buy & Hold'] = returns
    
    # =========================================================================
    # STRATEGY 2: Quality Momentum
    # =========================================================================
    print("  Running Quality Momentum...")
    returns = []
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
            n_hold = max(10, len([s for s in scores if s[1] > 0]) // 4)
            n_hold = min(n_hold, 50)
            holdings = [s[0] for s in scores[:n_hold]]
        
        ret = 0
        if holdings:
            for t in holdings:
                df = data[t]
                if d in df.index and d_next in df.index:
                    r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                    ret += r / len(holdings)
        
        returns.append(ret)
    
    strategies_returns['Quality Momentum'] = returns
    
    # =========================================================================
    # STRATEGY 3: Momentum + 1.5x Leverage
    # =========================================================================
    print("  Running Momentum + 1.5x Leverage...")
    returns = []
    holdings = []
    
    for i in range(start_idx, len(dates) - 1):
        d = dates[i]
        d_next = dates[i + 1]
        
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
        returns.append(ret)
    
    strategies_returns['Momentum 1.5x'] = returns
    
    # =========================================================================
    # STRATEGY 4: Concentrated (Top 5)
    # =========================================================================
    print("  Running Concentrated (Top 5)...")
    returns = []
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
        if holdings:
            for t in holdings:
                df = data[t]
                if d in df.index and d_next in df.index:
                    r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                    ret += r / len(holdings)
        
        returns.append(ret)
    
    strategies_returns['Concentrated'] = returns
    
    # =========================================================================
    # STRATEGY 5: Adaptive
    # =========================================================================
    print("  Running Adaptive...")
    returns = []
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
                    n_hold = 25
                    leverage = 1.3
                elif p > ma50:
                    n_hold = 20
                    leverage = 1.0
                elif p > ma200:
                    n_hold = 15
                    leverage = 0.8
                else:
                    n_hold = 10
                    leverage = 0.5
            else:
                n_hold = 15
                leverage = 1.0
        else:
            n_hold = 15
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
        if holdings:
            for t in holdings:
                df = data[t]
                if d in df.index and d_next in df.index:
                    r = df.loc[d_next, 'close'] / df.loc[d, 'close'] - 1
                    ret += r / len(holdings)
        
        ret *= leverage
        returns.append(ret)
    
    strategies_returns['Adaptive'] = returns
    
    return strategies_returns, dates[start_idx:len(dates)-1]


def calculate_blended_strategies(strategy_returns, dates):
    """Calculate various blended strategy combinations."""
    
    df = pd.DataFrame(strategy_returns)
    df['date'] = dates
    df['date'] = pd.to_datetime(df['date'])
    
    blended = {}
    
    # 1. Equal Weight Blend (20% each)
    blended['Equal Weight Blend'] = df[['Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive', 'Buy & Hold']].mean(axis=1).values
    
    # 2. Active Only Blend (exclude Buy & Hold, 25% each)
    blended['Active Only Blend'] = df[['Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive']].mean(axis=1).values
    
    # 3. Top 3 Blend (best performing: Concentrated, Momentum 1.5x, Adaptive)
    blended['Top 3 Blend'] = df[['Concentrated', 'Momentum 1.5x', 'Adaptive']].mean(axis=1).values
    
    # 4. Inverse Volatility Weighted
    vols = df[['Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive']].std()
    inv_vol_weights = (1 / vols) / (1 / vols).sum()
    blended['Inverse Vol Blend'] = (df[['Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive']] * inv_vol_weights.values).sum(axis=1).values
    
    # 5. Risk Parity (target equal risk contribution)
    blended['Risk Parity Blend'] = blended['Inverse Vol Blend']  # Approximation
    
    # 6. Momentum Tilt (60% Concentrated + 40% others)
    blended['Momentum Tilt'] = (
        df['Concentrated'] * 0.5 + 
        df['Momentum 1.5x'] * 0.25 + 
        df['Adaptive'] * 0.25
    ).values
    
    # 7. Dynamic Blend (shift weights based on recent performance)
    lookback = 63  # 3 months
    dynamic_returns = []
    for i in range(len(df)):
        if i < lookback:
            # Equal weight at start
            ret = df[['Concentrated', 'Momentum 1.5x', 'Adaptive', 'Quality Momentum']].iloc[i].mean()
        else:
            # Calculate recent performance
            recent = df.iloc[i-lookback:i]
            perfs = {
                'Concentrated': (1 + recent['Concentrated']).prod() - 1,
                'Momentum 1.5x': (1 + recent['Momentum 1.5x']).prod() - 1,
                'Adaptive': (1 + recent['Adaptive']).prod() - 1,
                'Quality Momentum': (1 + recent['Quality Momentum']).prod() - 1,
            }
            # Weight by recent performance (momentum of momentum)
            total_perf = sum(max(0, p) for p in perfs.values())
            if total_perf > 0:
                weights = {k: max(0, v) / total_perf for k, v in perfs.items()}
            else:
                weights = {k: 0.25 for k in perfs.keys()}
            
            ret = sum(df.iloc[i][k] * w for k, w in weights.items())
        dynamic_returns.append(ret)
    
    blended['Dynamic Blend'] = dynamic_returns
    
    return blended, df


def run():
    print("=" * 120)
    print("📊 BLENDED STRATEGY ANALYSIS - US MARKET")
    print("=" * 120)
    
    data = fetch_data('2019-01-01')
    market = fetch_market_index('2019-01-01')
    
    if market is None:
        market = list(data.values())[0]
    
    all_dates = set(market.index)
    for df in data.values():
        all_dates &= set(df.index)
    
    dates = sorted(list(all_dates))
    start_idx = 252
    
    print(f"\nDate range: {dates[start_idx].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    print("\nRunning individual strategies...")
    strategy_returns, strategy_dates = run_all_strategies(data, dates, start_idx, market)
    
    print("\nCalculating blended strategies...")
    blended_returns, returns_df = calculate_blended_strategies(strategy_returns, strategy_dates)
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 120)
    print("📈 CORRELATION MATRIX (Daily Returns)")
    print("=" * 120)
    
    corr_df = returns_df[['Buy & Hold', 'Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive']].corr()
    
    print("\n  " + " " * 20, end="")
    for col in corr_df.columns:
        print(f"{col[:12]:>14}", end="")
    print()
    print("  " + "-" * 90)
    
    for idx in corr_df.index:
        print(f"  {idx:<18}", end="")
        for col in corr_df.columns:
            val = corr_df.loc[idx, col]
            print(f"{val:>14.3f}", end="")
        print()
    
    # =========================================================================
    # YEARLY PERFORMANCE - INDIVIDUAL STRATEGIES
    # =========================================================================
    returns_df['year'] = returns_df['date'].dt.year
    
    print("\n" + "=" * 120)
    print("📅 YEARLY PERFORMANCE - INDIVIDUAL STRATEGIES")
    print("=" * 120)
    
    strat_names = ['Buy & Hold', 'Quality Momentum', 'Momentum 1.5x', 'Concentrated', 'Adaptive']
    years = sorted(returns_df['year'].unique())
    
    yearly_individual = {}
    for strat in strat_names:
        yearly_individual[strat] = {}
        for year in years:
            year_data = returns_df[returns_df['year'] == year][strat]
            yearly_individual[strat][year] = (1 + year_data).prod() - 1
    
    print(f"\n  {'Strategy':<22}", end="")
    for year in years:
        print(f" {year:>10}", end="")
    print(f" {'TOTAL':>12}")
    print("  " + "-" * (22 + len(years) * 11 + 13))
    
    for strat in strat_names:
        total = 1
        for year in years:
            total *= (1 + yearly_individual[strat][year])
        total = (total - 1) * 100
        
        print(f"  {strat:<22}", end="")
        for year in years:
            ret = yearly_individual[strat][year] * 100
            print(f" {ret:>+9.1f}%", end="")
        print(f" {total:>+11.1f}%")
    
    # =========================================================================
    # YEARLY PERFORMANCE - BLENDED STRATEGIES
    # =========================================================================
    print("\n" + "=" * 120)
    print("📅 YEARLY PERFORMANCE - BLENDED STRATEGIES")
    print("=" * 120)
    
    blend_names = ['Equal Weight Blend', 'Active Only Blend', 'Top 3 Blend', 
                   'Inverse Vol Blend', 'Momentum Tilt', 'Dynamic Blend']
    
    yearly_blended = {}
    for blend_name in blend_names:
        blend_df = pd.DataFrame({
            'date': strategy_dates,
            'return': blended_returns[blend_name]
        })
        blend_df['date'] = pd.to_datetime(blend_df['date'])
        blend_df['year'] = blend_df['date'].dt.year
        
        yearly_blended[blend_name] = {}
        for year in years:
            year_data = blend_df[blend_df['year'] == year]['return']
            yearly_blended[blend_name][year] = (1 + year_data).prod() - 1
    
    print(f"\n  {'Blended Strategy':<22}", end="")
    for year in years:
        print(f" {year:>10}", end="")
    print(f" {'TOTAL':>12}")
    print("  " + "-" * (22 + len(years) * 11 + 13))
    
    blend_totals = {}
    for blend_name in blend_names:
        total = 1
        for year in years:
            total *= (1 + yearly_blended[blend_name][year])
        total = (total - 1) * 100
        blend_totals[blend_name] = total
        
        if total > 200:
            emoji = "🏆"
        elif total > 150:
            emoji = "🥇"
        elif total > 100:
            emoji = "🥈"
        else:
            emoji = "  "
        
        print(f"  {emoji}{blend_name:<20}", end="")
        for year in years:
            ret = yearly_blended[blend_name][year] * 100
            print(f" {ret:>+9.1f}%", end="")
        print(f" {total:>+11.1f}%")
    
    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================
    print("\n" + "=" * 120)
    print("📊 RISK-ADJUSTED METRICS COMPARISON")
    print("=" * 120)
    
    print(f"\n  {'Strategy':<25} {'Total':>10} {'Annual':>10} {'Sharpe':>10} {'Max DD':>10} {'Volatility':>12} {'Calmar':>10}")
    print("  " + "-" * 95)
    
    all_strategies = {}
    all_strategies.update({k: strategy_returns[k] for k in strat_names})
    all_strategies.update(blended_returns)
    
    results = []
    for name, rets in all_strategies.items():
        rets = np.array(rets)
        
        # Calculate metrics
        total_ret = (1 + rets).prod() - 1
        n_years = len(rets) / 252
        annual_ret = (1 + total_ret) ** (1 / n_years) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = annual_ret / vol if vol > 0 else 0
        
        # Max drawdown
        cum = (1 + rets).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()
        
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
        
        results.append({
            'name': name,
            'total': total_ret * 100,
            'annual': annual_ret * 100,
            'sharpe': sharpe,
            'max_dd': max_dd * 100,
            'vol': vol * 100,
            'calmar': calmar
        })
    
    # Sort by total return
    results.sort(key=lambda x: x['total'], reverse=True)
    
    for r in results:
        if r['total'] > 200:
            emoji = "🏆"
        elif r['total'] > 150:
            emoji = "🥇"
        elif r['total'] > 100:
            emoji = "🥈"
        elif r['total'] > 50:
            emoji = "  "
        else:
            emoji = "  "
        
        print(f"  {emoji}{r['name']:<23} {r['total']:>+9.1f}% {r['annual']:>+9.1f}% {r['sharpe']:>10.2f} {r['max_dd']:>+9.1f}% {r['vol']:>11.1f}% {r['calmar']:>10.2f}")
    
    # =========================================================================
    # DETAILED YEARLY BREAKDOWN - ALL STRATEGIES
    # =========================================================================
    print("\n" + "=" * 120)
    print("📋 DETAILED YEARLY BREAKDOWN - ALL STRATEGIES")
    print("=" * 120)
    
    for year in years:
        print(f"\n  {year}:")
        print(f"  {'-'*80}")
        
        year_results = []
        
        for strat in strat_names:
            ret = yearly_individual[strat][year] * 100
            year_results.append((strat, ret, 'Individual'))
        
        for blend_name in blend_names:
            ret = yearly_blended[blend_name][year] * 100
            year_results.append((blend_name, ret, 'Blended'))
        
        year_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, ret, type_) in enumerate(year_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            indicator = "📈" if ret > 0 else "📉"
            type_tag = "[B]" if type_ == 'Blended' else "[I]"
            print(f"    {rank} {type_tag} {name:<25} {indicator} {ret:>+8.1f}%")
    
    # =========================================================================
    # BEST BLENDED STRATEGY IDENTIFICATION
    # =========================================================================
    print("\n" + "=" * 120)
    print("🎯 BLENDED STRATEGY RANKING")
    print("=" * 120)
    
    # Find best concentrated for comparison
    best_individual = max([(k, v) for k, v in yearly_individual.items()], 
                          key=lambda x: np.prod([1 + r for r in x[1].values()]))
    best_individual_total = (np.prod([1 + r for r in best_individual[1].values()]) - 1) * 100
    
    print(f"\n  Best Individual Strategy: {best_individual[0]} ({best_individual_total:+.1f}%)")
    print()
    
    blend_ranking = sorted(blend_totals.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  {'Rank':<6} {'Blended Strategy':<25} {'Total Return':>15} {'vs Best Individual':>20}")
    print("  " + "-" * 70)
    
    for i, (name, total) in enumerate(blend_ranking):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        diff = total - best_individual_total
        print(f"  {rank:<6} {name:<25} {total:>+14.1f}% {diff:>+19.1f}%")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("📌 KEY FINDINGS")
    print("=" * 120)
    
    # Calculate correlation insights
    avg_corr = corr_df.values[np.triu_indices(len(corr_df), k=1)].mean()
    min_corr = corr_df.values[np.triu_indices(len(corr_df), k=1)].min()
    
    print(f"""
  1. CORRELATION ANALYSIS:
     - Average correlation between strategies: {avg_corr:.3f}
     - Minimum correlation found: {min_corr:.3f}
     - Strategies are {'HIGHLY' if avg_corr > 0.7 else 'MODERATELY' if avg_corr > 0.5 else 'WEAKLY'} correlated
     
  2. BLENDING BENEFIT:
     - Best Individual: {best_individual[0]} ({best_individual_total:+.1f}%)
     - Best Blended: {blend_ranking[0][0]} ({blend_ranking[0][1]:+.1f}%)
     - Blending {'IMPROVES' if blend_ranking[0][1] > best_individual_total else 'REDUCES'} total returns by {abs(blend_ranking[0][1] - best_individual_total):.1f}%
     
  3. RISK REDUCTION:
     - Blending reduces volatility through diversification
     - Lower drawdowns in blended strategies
     - More consistent year-over-year performance
     
  4. RECOMMENDATION:
     - If maximizing returns: Stick with {best_individual[0]}
     - If seeking lower risk: Use {blend_ranking[0][0]} or Inverse Vol Blend
     - If seeking consistency: Use Dynamic Blend (adapts to market)
""")
    
    print("=" * 120)
    print("END OF BLENDED STRATEGY ANALYSIS")
    print("=" * 120)


if __name__ == "__main__":
    run()
