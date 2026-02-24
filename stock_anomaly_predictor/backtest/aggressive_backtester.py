#!/usr/bin/env python3
"""
AGGRESSIVE STRATEGY BACKTESTER
==============================
Push for maximum returns with:
1. Higher leverage (2x-3x)
2. More concentrated positions (3-5 stocks)
3. Faster rotation (daily/weekly)
4. Momentum factor amplification
5. Risk parity sizing
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
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
            if len(df) >= 250:
                df.columns = [c.lower() for c in df.columns]
                df.index = df.index.tz_localize(None)
                data[ticker] = df
        except:
            pass
    
    print(f"Loaded {len(data)} stocks")
    return data


def calculate_momentum_score(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Calculate momentum score (higher = better)."""
    close = df['close']
    
    # Multi-factor momentum
    mom_20 = close.pct_change(20)
    mom_60 = close.pct_change(60)
    mom_120 = close.pct_change(120) if len(close) > 120 else mom_60
    
    # Weighted momentum
    score = mom_20 * 0.5 + mom_60 * 0.3 + mom_120 * 0.2
    
    # Bonus for being above MAs
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean() if len(close) >= 200 else ma_50
    
    above_ma = (close > ma_50).astype(float) * 0.1 + (close > ma_200).astype(float) * 0.1
    
    return score + above_ma


def run_aggressive_backtest():
    """Run aggressive backtest strategies."""
    
    print("=" * 80)
    print("AGGRESSIVE STRATEGY BACKTESTER")
    print("=" * 80)
    
    data = fetch_data('2019-01-01')
    
    # Find common trading dates
    all_dates = None
    for ticker, df in data.items():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))
    
    dates = sorted(list(all_dates))
    start_idx = 250
    
    if len(dates) < start_idx + 100:
        print("Insufficient data")
        return
    
    # Pre-calculate momentum for all stocks
    print("\nCalculating momentum scores...")
    momentum_scores = {}
    for ticker, df in data.items():
        momentum_scores[ticker] = calculate_momentum_score(df)
    
    # Market regime detection (using Aramco)
    market = data.get('2222.SR', list(data.values())[0])
    market_ma50 = market['close'].rolling(50).mean()
    market_ma200 = market['close'].rolling(200).mean()
    
    # Run multiple strategies
    strategies = {}
    
    # =========================================================================
    # STRATEGY 1: Ultra Concentrated + High Leverage
    # =========================================================================
    print("\nRunning Strategy 1: Ultra Concentrated (3 stocks, 3x leverage)...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Get momentum scores for this date
        scores = []
        for ticker in data.keys():
            if current_date in momentum_scores[ticker].index:
                score = momentum_scores[ticker].loc[current_date]
                if not np.isnan(score):
                    scores.append((ticker, score))
        
        # Top 3 stocks
        scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = [s[0] for s in scores[:3]]
        
        # Market regime
        if current_date in market.index:
            idx = market.index.get_loc(current_date)
            if idx >= 50:
                price = market['close'].iloc[idx]
                ma50 = market_ma50.iloc[idx]
                ma200 = market_ma200.iloc[idx] if idx >= 200 else ma50
                
                if price > ma50 > ma200:
                    leverage = 3.0
                elif price > ma50:
                    leverage = 2.0
                elif price > ma200:
                    leverage = 1.0
                else:
                    leverage = 0.5
            else:
                leverage = 1.0
        else:
            leverage = 1.0
        
        # Calculate return
        returns = []
        for ticker in top_3:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                returns.append(ret)
        
        if returns:
            avg_ret = np.mean(returns) * leverage
            # Cap extreme returns
            avg_ret = max(-0.15, min(0.15, avg_ret))
            portfolio.append(portfolio[-1] * (1 + avg_ret))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Ultra Concentrated (3x)'] = portfolio
    
    # =========================================================================
    # STRATEGY 2: Daily Rotation + Momentum Filter
    # =========================================================================
    print("Running Strategy 2: Daily Rotation with strict momentum filter...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Only stocks with positive momentum
        scores = []
        for ticker in data.keys():
            if current_date in momentum_scores[ticker].index:
                score = momentum_scores[ticker].loc[current_date]
                if not np.isnan(score) and score > 0:  # Positive momentum only
                    scores.append((ticker, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in scores[:5]]
        
        # Leverage based on # of qualifying stocks
        leverage = min(2.5, 1 + len(scores) / 20)
        
        # Calculate return
        returns = []
        for ticker in top_stocks:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                returns.append(ret)
        
        if returns:
            avg_ret = np.mean(returns) * leverage
            avg_ret = max(-0.10, min(0.10, avg_ret))
            portfolio.append(portfolio[-1] * (1 + avg_ret))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Daily Rotation (Mom Filter)'] = portfolio
    
    # =========================================================================
    # STRATEGY 3: Momentum Breakout
    # =========================================================================
    print("Running Strategy 3: Momentum Breakout (buy 52-week highs)...")
    
    portfolio = [100]
    holdings = {}  # ticker -> entry_price
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Find stocks at 52-week highs
        new_highs = []
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                if idx >= 252:
                    current_price = df['close'].iloc[idx]
                    high_52w = df['high'].iloc[idx-252:idx].max()
                    
                    if current_price >= high_52w * 0.98:  # Within 2% of high
                        mom = momentum_scores[ticker].loc[current_date] if current_date in momentum_scores[ticker].index else 0
                        if not np.isnan(mom):
                            new_highs.append((ticker, mom))
        
        new_highs.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in new_highs[:7]]
        
        # Leverage
        leverage = 2.0 if len(new_highs) > 10 else 1.5 if len(new_highs) > 5 else 1.0
        
        returns = []
        for ticker in top_stocks:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                returns.append(ret)
        
        if returns:
            avg_ret = np.mean(returns) * leverage
            avg_ret = max(-0.12, min(0.12, avg_ret))
            portfolio.append(portfolio[-1] * (1 + avg_ret))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Momentum Breakout'] = portfolio
    
    # =========================================================================
    # STRATEGY 4: Risk Parity + Momentum
    # =========================================================================
    print("Running Strategy 4: Risk Parity weighted by momentum...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        # Get stocks with momentum and volatility
        candidates = []
        for ticker, df in data.items():
            if current_date in df.index and current_date in momentum_scores[ticker].index:
                idx = df.index.get_loc(current_date)
                if idx >= 20:
                    mom = momentum_scores[ticker].loc[current_date]
                    vol = df['close'].pct_change().iloc[idx-20:idx].std()
                    
                    if not np.isnan(mom) and not np.isnan(vol) and vol > 0 and mom > 0:
                        # Risk-adjusted momentum
                        risk_adj_mom = mom / vol
                        candidates.append((ticker, risk_adj_mom, vol))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_stocks = candidates[:10]
        
        if top_stocks:
            # Inverse volatility weighting
            total_inv_vol = sum(1/c[2] for c in top_stocks)
            weights = {c[0]: (1/c[2]) / total_inv_vol for c in top_stocks}
            
            # Market regime leverage
            if current_date in market.index:
                idx = market.index.get_loc(current_date)
                if idx >= 50:
                    price = market['close'].iloc[idx]
                    ma50 = market_ma50.iloc[idx]
                    leverage = 2.0 if price > ma50 else 1.0
                else:
                    leverage = 1.0
            else:
                leverage = 1.0
            
            # Weighted return
            port_ret = 0
            for ticker, weight in weights.items():
                df = data[ticker]
                if next_date in df.index:
                    ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                    port_ret += ret * weight
            
            port_ret *= leverage
            port_ret = max(-0.08, min(0.08, port_ret))
            portfolio.append(portfolio[-1] * (1 + port_ret))
        else:
            portfolio.append(portfolio[-1])
    
    strategies['Risk Parity + Momentum'] = portfolio
    
    # =========================================================================
    # STRATEGY 5: Pure Trend Following
    # =========================================================================
    print("Running Strategy 5: Pure Trend Following (long/short)...")
    
    portfolio = [100]
    
    for i in range(start_idx, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        
        long_stocks = []
        short_stocks = []
        
        for ticker, df in data.items():
            if current_date in df.index:
                idx = df.index.get_loc(current_date)
                if idx >= 200:
                    price = df['close'].iloc[idx]
                    ma_50 = df['close'].iloc[idx-50:idx].mean()
                    ma_200 = df['close'].iloc[idx-200:idx].mean()
                    mom = momentum_scores[ticker].loc[current_date] if current_date in momentum_scores[ticker].index else 0
                    
                    if price > ma_50 > ma_200 and mom > 0:
                        long_stocks.append((ticker, mom))
                    elif price < ma_50 < ma_200 and mom < 0:
                        short_stocks.append((ticker, -mom))
        
        long_stocks.sort(key=lambda x: x[1], reverse=True)
        short_stocks.sort(key=lambda x: x[1], reverse=True)
        
        top_long = [s[0] for s in long_stocks[:5]]
        top_short = [s[0] for s in short_stocks[:3]]
        
        long_ret = 0
        short_ret = 0
        
        for ticker in top_long:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                long_ret += ret / max(1, len(top_long))
        
        for ticker in top_short:
            df = data[ticker]
            if current_date in df.index and next_date in df.index:
                ret = df.loc[next_date, 'close'] / df.loc[current_date, 'close'] - 1
                short_ret -= ret / max(1, len(top_short))  # Negative for short
        
        # Combined (70% long, 30% short exposure)
        total_ret = long_ret * 1.5 + short_ret * 0.5
        total_ret = max(-0.10, min(0.10, total_ret))
        portfolio.append(portfolio[-1] * (1 + total_ret))
    
    strategies['Trend Following L/S'] = portfolio
    
    # =========================================================================
    # BASELINE: Buy and Hold
    # =========================================================================
    print("Running Baseline: Buy and Hold...")
    
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
    print("📊 AGGRESSIVE BACKTEST RESULTS (2019-2026)")
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
        
        excess = total_ret - baseline_return
        
        results.append({
            'name': name,
            'total': total_ret,
            'annual': annual,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'excess': excess
        })
    
    results.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n  {'Strategy':<35} {'Total%':>10} {'Annual%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'vs B&H':>10}")
    print("  " + "-" * 85)
    
    for r in results:
        emoji = "🏆" if r['excess'] > 100 else "🥇" if r['excess'] > 50 else "✅" if r['excess'] > 0 else "❌"
        print(f"  {emoji} {r['name']:<33} {r['total']:>+9.1f}% {r['annual']:>+9.1f}% "
              f"{r['sharpe']:>7.2f} {r['max_dd']:>7.1f}% {r['excess']:>+9.1f}%")
    
    # Winner
    winner = results[0]
    
    print(f"""

{'='*100}
🏆 CHAMPION STRATEGY: {winner['name']}
{'='*100}

  PERFORMANCE:
    Total Return:      {winner['total']:+.1f}%
    Annual Return:     {winner['annual']:+.1f}%
    Sharpe Ratio:      {winner['sharpe']:.2f}
    Max Drawdown:      {winner['max_dd']:.1f}%
  
  VS BUY & HOLD:
    Excess Return:     {winner['excess']:+.1f}%
    Outperformance:    {winner['excess']/max(1, abs(baseline_return)):.1f}x

{'='*100}
""")
    
    return results


if __name__ == "__main__":
    run_aggressive_backtest()
