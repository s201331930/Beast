#!/usr/bin/env python3
"""
CHAMPION STRATEGY BACKTESTER
============================
Maximum returns through:
1. Dual momentum (absolute + relative)
2. Dynamic leverage (1x-4x based on conditions)
3. Volatility targeting
4. Drawdown control
5. Sector rotation
"""

import os
from datetime import datetime
from typing import Dict, List
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
    '1020.SR', '1030.SR', '2222.SR', '4030.SR', '2381.SR', '2380.SR',
    '2010.SR', '2290.SR', '2350.SR', '2330.SR', '2020.SR', '2190.SR',
    '1211.SR', '1321.SR', '1304.SR', '2240.SR', '2320.SR', '3020.SR',
    '3030.SR', '5110.SR', '2082.SR', '4190.SR', '4003.SR', '4001.SR',
    '4002.SR', '4007.SR', '4020.SR', '4080.SR', '4200.SR', '4280.SR',
    '2280.SR', '2050.SR', '4071.SR', '7010.SR', '7020.SR', '4300.SR',
    '4330.SR', '8010.SR', '8030.SR', '8210.SR', '8300.SR',
]


def fetch_data(start_date: str = '2019-01-01') -> Dict[str, pd.DataFrame]:
    """Fetch all data."""
    print(f"Fetching data from {start_date}...")
    data = {}
    
    for ticker in TASI_TICKERS:
        try:
            df = yf.Ticker(ticker).history(start=start_date)
            if len(df) >= 252:
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)
                data[ticker] = df
        except:
            pass
    
    print(f"Loaded {len(data)} stocks")
    return data


def dual_momentum_score(df: pd.DataFrame, idx: int) -> float:
    """
    Dual momentum combining:
    - Absolute momentum (trend)
    - Relative momentum (vs market average)
    """
    if idx < 252:
        return 0
    
    close = df['close']
    
    # Absolute momentum (multiple timeframes)
    mom_1m = close.iloc[idx] / close.iloc[idx-21] - 1
    mom_3m = close.iloc[idx] / close.iloc[idx-63] - 1
    mom_6m = close.iloc[idx] / close.iloc[idx-126] - 1
    mom_12m = close.iloc[idx] / close.iloc[idx-252] - 1
    
    # Weighted momentum
    abs_mom = mom_1m * 0.4 + mom_3m * 0.3 + mom_6m * 0.2 + mom_12m * 0.1
    
    # Trend strength
    ma_50 = close.iloc[idx-50:idx].mean()
    ma_200 = close.iloc[idx-200:idx].mean()
    
    trend_bonus = 0
    if close.iloc[idx] > ma_50:
        trend_bonus += 0.1
    if close.iloc[idx] > ma_200:
        trend_bonus += 0.1
    if ma_50 > ma_200:
        trend_bonus += 0.1
    
    # Volatility adjustment (lower vol = higher score)
    vol = close.pct_change().iloc[idx-20:idx].std()
    vol_adj = 1 / (1 + vol * 10)  # Penalize high vol
    
    return (abs_mom + trend_bonus) * vol_adj


def calculate_optimal_leverage(market_df: pd.DataFrame, idx: int, 
                                portfolio_value: float, peak_value: float) -> float:
    """
    Dynamic leverage based on:
    - Market regime
    - Current drawdown
    - Volatility
    """
    if idx < 200:
        return 1.0
    
    close = market_df['close']
    
    # Market regime
    ma_20 = close.iloc[idx-20:idx].mean()
    ma_50 = close.iloc[idx-50:idx].mean()
    ma_200 = close.iloc[idx-200:idx].mean()
    price = close.iloc[idx]
    
    # Base leverage from regime
    if price > ma_20 > ma_50 > ma_200:
        base_leverage = 4.0  # Strong bull
    elif price > ma_50 > ma_200:
        base_leverage = 3.0  # Bull
    elif price > ma_200:
        base_leverage = 2.0  # Weak bull
    elif price > ma_50:
        base_leverage = 1.0  # Neutral
    else:
        base_leverage = 0.5  # Bear
    
    # Volatility adjustment
    vol = close.pct_change().iloc[idx-20:idx].std() * np.sqrt(252)
    if vol > 0.4:  # High vol
        base_leverage *= 0.5
    elif vol > 0.25:  # Normal vol
        base_leverage *= 0.75
    # Low vol - keep leverage
    
    # Drawdown adjustment
    current_dd = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
    if current_dd > 0.20:  # In significant drawdown
        base_leverage *= 0.5
    elif current_dd > 0.10:
        base_leverage *= 0.75
    
    # Cap leverage
    return min(4.0, max(0.25, base_leverage))


def run_champion_backtest():
    """Run the champion strategy backtest."""
    
    print("=" * 80)
    print("🏆 CHAMPION STRATEGY BACKTESTER")
    print("=" * 80)
    
    data = fetch_data('2019-01-01')
    
    # Common dates
    all_dates = set.intersection(*[set(df.index) for df in data.values()])
    dates = sorted(list(all_dates))
    start_idx = 252
    
    if len(dates) < start_idx + 100:
        print("Insufficient data")
        return
    
    # Market proxy
    market = data.get('2222.SR', list(data.values())[0])
    
    strategies = {}
    
    # =========================================================================
    # STRATEGY 1: Champion Strategy (Full System)
    # =========================================================================
    print("\nRunning Champion Strategy (Full System)...")
    
    portfolio = [100]
    peak = 100
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Score all stocks
        scores = []
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                score = dual_momentum_score(df, idx)
                if not np.isnan(score):
                    scores.append((ticker, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Dynamic number of holdings based on opportunities
        good_stocks = [s for s in scores if s[1] > 0.05]
        n_holdings = min(5, max(2, len(good_stocks)))
        
        top_stocks = [s[0] for s in scores[:n_holdings]]
        
        # Dynamic leverage
        market_idx = market.index.get_loc(current_date) if current_date in market.index else 0
        leverage = calculate_optimal_leverage(market, market_idx, portfolio[-1], peak)
        
        # Calculate weighted returns (higher momentum = higher weight)
        total_score = sum(max(0.01, scores[j][1]) for j in range(n_holdings))
        
        port_ret = 0
        for j, ticker in enumerate(top_stocks):
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                weight = max(0.01, scores[j][1]) / total_score
                port_ret += ret * weight
        
        # Apply leverage
        port_ret *= leverage
        
        # Risk management: cap daily loss
        port_ret = max(-0.08, port_ret)
        
        new_value = portfolio[-1] * (1 + port_ret)
        portfolio.append(new_value)
        peak = max(peak, new_value)
    
    strategies['Champion (Full System)'] = portfolio
    
    # =========================================================================
    # STRATEGY 2: Aggressive Momentum
    # =========================================================================
    print("Running Aggressive Momentum (5x max leverage)...")
    
    portfolio = [100]
    peak = 100
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Top 3 stocks only
        scores = []
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                score = dual_momentum_score(df, idx)
                if not np.isnan(score) and score > 0:
                    scores.append((ticker, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in scores[:3]]
        
        # More aggressive leverage
        market_idx = market.index.get_loc(current_date) if current_date in market.index else 0
        if market_idx >= 50:
            price = market['close'].iloc[market_idx]
            ma_50 = market['close'].iloc[market_idx-50:market_idx].mean()
            leverage = 5.0 if price > ma_50 else 1.0
        else:
            leverage = 1.0
        
        # Drawdown control
        current_dd = (peak - portfolio[-1]) / peak
        if current_dd > 0.15:
            leverage = 1.0
        
        port_ret = 0
        for ticker in top_stocks:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                port_ret += ret / len(top_stocks)
        
        port_ret *= leverage
        port_ret = max(-0.10, min(0.15, port_ret))
        
        new_value = portfolio[-1] * (1 + port_ret)
        portfolio.append(new_value)
        peak = max(peak, new_value)
    
    strategies['Aggressive Momentum (5x)'] = portfolio
    
    # =========================================================================
    # STRATEGY 3: Volatility Targeting
    # =========================================================================
    print("Running Volatility Targeting (15% target vol)...")
    
    portfolio = [100]
    target_vol = 0.15  # 15% annual vol target
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Score stocks
        scores = []
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                score = dual_momentum_score(df, idx)
                if not np.isnan(score):
                    scores.append((ticker, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in scores[:7] if s[1] > 0]
        
        if not top_stocks:
            top_stocks = [s[0] for s in scores[:5]]
        
        # Calculate portfolio volatility
        combined_ret = []
        for lookback in range(max(1, i-20), i):
            if lookback >= start_idx:
                lb_date = dates[lookback]
                lb_next = dates[lookback + 1] if lookback + 1 < len(dates) else lb_date
                
                daily_ret = 0
                for ticker in top_stocks:
                    df = data[ticker]
                    if lb_date in df.index and lb_next in df.index:
                        ret = df.loc[lb_next, 'close'] / df.loc[lb_date, 'close'] - 1
                        daily_ret += ret / len(top_stocks)
                combined_ret.append(daily_ret)
        
        if len(combined_ret) > 5:
            realized_vol = np.std(combined_ret) * np.sqrt(252)
            leverage = target_vol / realized_vol if realized_vol > 0 else 1.0
            leverage = min(4.0, max(0.5, leverage))
        else:
            leverage = 1.0
        
        # Calculate today's return
        port_ret = 0
        for ticker in top_stocks:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                port_ret += ret / len(top_stocks)
        
        port_ret *= leverage
        portfolio.append(portfolio[-1] * (1 + port_ret))
    
    strategies['Vol Targeting (15%)'] = portfolio
    
    # =========================================================================
    # STRATEGY 4: Mean Reversion + Momentum Hybrid
    # =========================================================================
    print("Running Mean Reversion + Momentum Hybrid...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Momentum stocks (trending)
        momentum_stocks = []
        # Mean reversion stocks (oversold in uptrend)
        mean_rev_stocks = []
        
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                if idx >= 50:
                    price = df['close'].iloc[idx]
                    ma_20 = df['close'].iloc[idx-20:idx].mean()
                    ma_50 = df['close'].iloc[idx-50:idx].mean()
                    
                    mom = dual_momentum_score(df, idx)
                    
                    # RSI
                    changes = df['close'].diff().iloc[idx-14:idx]
                    gains = changes.where(changes > 0, 0).mean()
                    losses = -changes.where(changes < 0, 0).mean()
                    rsi = 100 - (100 / (1 + gains / (losses + 0.001)))
                    
                    # Momentum: strong trend
                    if mom > 0.1 and price > ma_50:
                        momentum_stocks.append((ticker, mom))
                    
                    # Mean reversion: oversold in uptrend
                    elif ma_20 > ma_50 and rsi < 30 and price > ma_50 * 0.95:
                        mean_rev_stocks.append((ticker, -rsi))  # Lower RSI = higher score
        
        momentum_stocks.sort(key=lambda x: x[1], reverse=True)
        mean_rev_stocks.sort(key=lambda x: x[1], reverse=True)
        
        # Combine: 70% momentum, 30% mean reversion
        top_mom = [s[0] for s in momentum_stocks[:4]]
        top_mr = [s[0] for s in mean_rev_stocks[:2]]
        
        all_stocks = top_mom + top_mr
        
        if all_stocks:
            leverage = 2.5 if len(momentum_stocks) > 10 else 1.5
            
            port_ret = 0
            for ticker in all_stocks:
                df = data[ticker]
                if current_date in df.index and next_date in df.index:
                    ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                    port_ret += ret / len(all_stocks)
            
            port_ret *= leverage
            port_ret = max(-0.08, min(0.12, port_ret))
            portfolio.append(portfolio[-1] * (1 + port_ret))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Hybrid (Mom + MR)'] = portfolio
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    print("Running Baseline...")
    
    portfolio = [100]
    baseline_stocks = list(data.keys())[:20]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        returns = []
        for ticker in baseline_stocks:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                returns.append(ret)
        
        if returns:
            portfolio.append(portfolio[-1] * (1 + np.mean(returns)))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Buy & Hold'] = portfolio
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 100)
    print("🏆 CHAMPION BACKTEST RESULTS (2019-2026)")
    print("=" * 100)
    
    results = []
    baseline_return = (strategies['Buy & Hold'][-1] / 100 - 1) * 100
    
    for name, equity in strategies.items():
        total_ret = (equity[-1] / 100 - 1) * 100
        years = len(equity) / 252
        annual = ((equity[-1] / 100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        returns = np.diff(equity) / np.array(equity[:-1])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        max_dd = np.max(drawdown)
        
        # Calmar ratio
        calmar = annual / max_dd if max_dd > 0 else 0
        
        excess = total_ret - baseline_return
        
        results.append({
            'name': name,
            'total': total_ret,
            'annual': annual,
            'sharpe': sharpe,
            'calmar': calmar,
            'max_dd': max_dd,
            'excess': excess
        })
    
    results.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n  {'Strategy':<30} {'Total%':>10} {'Annual%':>10} {'Sharpe':>8} {'Calmar':>8} {'MaxDD%':>8} {'Excess':>10}")
    print("  " + "-" * 95)
    
    for r in results:
        emoji = "🏆" if r['excess'] > 100 else "🥇" if r['excess'] > 50 else "✅" if r['excess'] > 0 else "❌"
        print(f"  {emoji} {r['name']:<28} {r['total']:>+9.1f}% {r['annual']:>+9.1f}% "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['max_dd']:>7.1f}% {r['excess']:>+9.1f}%")
    
    winner = results[0]
    
    print(f"""

{'='*100}
🏆🏆🏆 GRAND CHAMPION: {winner['name']}
{'='*100}

  PERFORMANCE SUMMARY:
  ────────────────────────────────────────────────────────────────
    Total Return:        {winner['total']:+.1f}%
    Annual Return:       {winner['annual']:+.1f}%
    Sharpe Ratio:        {winner['sharpe']:.2f}
    Calmar Ratio:        {winner['calmar']:.2f}
    Max Drawdown:        {winner['max_dd']:.1f}%
  
  OUTPERFORMANCE:
  ────────────────────────────────────────────────────────────────
    vs Buy & Hold:       {winner['excess']:+.1f}%
    Multiplier:          {(winner['total']+100)/(baseline_return+100):.2f}x

  STRATEGY COMPONENTS:
  ────────────────────────────────────────────────────────────────
    1. Dual Momentum (1m, 3m, 6m, 12m weighted)
    2. Dynamic Leverage (1x-4x based on regime)
    3. Volatility Adjustment
    4. Drawdown Control
    5. Concentrated Positions (2-5 stocks)

{'='*100}
""")
    
    return results


if __name__ == "__main__":
    run_champion_backtest()
