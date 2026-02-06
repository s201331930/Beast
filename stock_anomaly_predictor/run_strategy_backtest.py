#!/usr/bin/env python3
"""
STRATEGY BACKTEST
=================
Backtest the production strategy to get performance metrics.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from production.config import STRATEGY, TASI_STOCKS

print("=" * 80)
print("STRATEGY BACKTEST - Performance Metrics")
print("=" * 80)

# Load data for all stocks
STOCKS = list(TASI_STOCKS.keys())

print(f"\n[1] LOADING DATA FOR {len(STOCKS)} STOCKS...")
stock_data = {}
for ticker in STOCKS:
    try:
        df = yf.Ticker(ticker).history(start="2022-01-01")
        if len(df) >= 100:
            df.columns = [c.lower() for c in df.columns]
            stock_data[ticker] = df
    except:
        pass

print(f"  Loaded: {len(stock_data)} stocks")

# Calculate indicators for all stocks
print("\n[2] CALCULATING INDICATORS...")

def calc_indicators(df):
    close = df['close']
    volume = df['volume']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    # Moving averages
    ma_50 = close.rolling(50).mean()
    
    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (close - bb_lower) / (4 * bb_std + 1e-10)
    
    # Volume ratio
    vol_ratio = volume / volume.rolling(20).mean()
    
    # Momentum
    momentum = close / close.shift(20) - 1
    
    # Volatility
    volatility = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    return {
        'rsi': rsi,
        'ma_50': ma_50,
        'bb_pos': bb_pos,
        'vol_ratio': vol_ratio,
        'momentum': momentum,
        'volatility': volatility
    }

indicators = {ticker: calc_indicators(df) for ticker, df in stock_data.items()}
print("  Done")

# Get common trading dates
all_dates = set()
for df in stock_data.values():
    all_dates.update(df.index.tolist())
trading_dates = sorted(all_dates)

print(f"  Trading period: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
print(f"  Total days: {len(trading_dates)}")

# Run backtest
print("\n[3] RUNNING BACKTEST...")

# Strategy parameters
TP_PCT = STRATEGY['take_profit_pct']
TRAIL_PCT = STRATEGY['trailing_stop_pct']
TRAIL_ACTIVATE = STRATEGY['trailing_activation_pct']
MAX_DAYS = STRATEGY['max_holding_days']
RSI_OVERSOLD = STRATEGY['rsi_oversold']
VOL_RATIO_THRESH = STRATEGY['volume_ratio_threshold']
MOM_THRESH = STRATEGY['momentum_threshold']
LEV_BULL = STRATEGY['leverage_bull']
LEV_BEAR = STRATEGY['leverage_bear']

capital = 1_000_000
cash = capital
positions = {}
trades = []
equity_curve = []

# Market regime tracker (using Al Rajhi as proxy)
market_ticker = '1180.SR'

for i in range(60, len(trading_dates) - 1):
    date = trading_dates[i]
    next_date = trading_dates[i + 1]
    
    # Get current prices
    current_prices = {}
    next_prices = {}
    for ticker, df in stock_data.items():
        if date in df.index:
            current_prices[ticker] = df.loc[date, 'close']
        if next_date in df.index:
            next_prices[ticker] = df.loc[next_date, 'close']
    
    # Detect market regime
    if market_ticker in stock_data and date in stock_data[market_ticker].index:
        market_price = current_prices.get(market_ticker, 0)
        market_ma50 = indicators[market_ticker]['ma_50'].loc[date] if date in indicators[market_ticker]['ma_50'].index else market_price
        regime = "BULL" if market_price > market_ma50 else "BEAR"
        leverage = LEV_BULL if regime == "BULL" else LEV_BEAR
    else:
        regime = "NEUTRAL"
        leverage = 1.0
    
    # Check exits
    for ticker in list(positions.keys()):
        if ticker not in current_prices:
            continue
        
        pos = positions[ticker]
        price = current_prices[ticker]
        entry = pos['entry']
        days_held = (date - pos['entry_date']).days
        current_return = price / entry - 1
        
        # Update high water mark
        if price > pos['high']:
            pos['high'] = price
        
        exit_reason = None
        
        # Take profit
        if current_return >= TP_PCT:
            exit_reason = 'take_profit'
        
        # Trailing stop (only after activation)
        elif pos['high'] > entry * (1 + TRAIL_ACTIVATE):
            trail_level = pos['high'] * (1 - TRAIL_PCT)
            if price <= trail_level:
                exit_reason = 'trailing_stop'
        
        # Time exit
        elif days_held >= MAX_DAYS:
            exit_reason = 'time_exit'
        
        if exit_reason and ticker in next_prices:
            exit_price = next_prices[ticker] * 0.999  # Slippage
            pnl = (exit_price - entry) * pos['shares']
            pnl_pct = exit_price / entry - 1
            
            trades.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': next_date,
                'entry_price': entry,
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'holding_days': days_held,
                'exit_reason': exit_reason,
                'regime': pos['regime']
            })
            
            cash += exit_price * pos['shares'] * 0.999
            del positions[ticker]
    
    # Check entries
    for ticker in stock_data.keys():
        if ticker in positions or len(positions) >= 15:
            continue
        if ticker not in current_prices or ticker not in next_prices:
            continue
        if ticker not in indicators:
            continue
        
        ind = indicators[ticker]
        if date not in ind['rsi'].index:
            continue
        
        rsi = ind['rsi'].loc[date]
        bb_pos = ind['bb_pos'].loc[date] if date in ind['bb_pos'].index else 0.5
        vol_ratio = ind['vol_ratio'].loc[date] if date in ind['vol_ratio'].index else 1.0
        momentum = ind['momentum'].loc[date] if date in ind['momentum'].index else 0
        ma_50 = ind['ma_50'].loc[date] if date in ind['ma_50'].index else current_prices[ticker]
        price = current_prices[ticker]
        
        # Check signals
        oversold_signal = rsi < RSI_OVERSOLD and bb_pos < 0.3 and vol_ratio > VOL_RATIO_THRESH
        momentum_signal = momentum > MOM_THRESH and price > ma_50 and rsi < 60
        
        if oversold_signal or momentum_signal:
            entry_price = next_prices[ticker] * 1.001  # Slippage
            
            # Position size based on leverage
            pos_value = cash * 0.07 * leverage
            shares = int(pos_value / entry_price)
            
            if shares > 0 and entry_price * shares < cash * 0.95:
                positions[ticker] = {
                    'entry_date': next_date,
                    'entry': entry_price,
                    'shares': shares,
                    'high': entry_price,
                    'regime': regime,
                    'signal': 'oversold' if oversold_signal else 'momentum'
                }
                cash -= entry_price * shares * 1.001
    
    # Record equity
    pos_value = sum(pos['shares'] * current_prices.get(t, pos['entry']) for t, pos in positions.items())
    equity_curve.append({
        'date': date,
        'cash': cash,
        'positions': pos_value,
        'total': cash + pos_value,
        'n_positions': len(positions),
        'regime': regime
    })

# Close remaining positions
print("\n[4] CLOSING REMAINING POSITIONS...")
for ticker, pos in list(positions.items()):
    if ticker in stock_data:
        final_price = stock_data[ticker]['close'].iloc[-1]
        pnl = (final_price - pos['entry']) * pos['shares']
        trades.append({
            'ticker': ticker,
            'entry_date': pos['entry_date'],
            'exit_date': trading_dates[-1],
            'entry_price': pos['entry'],
            'exit_price': final_price,
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_pct': final_price / pos['entry'] - 1,
            'holding_days': (trading_dates[-1] - pos['entry_date']).days,
            'exit_reason': 'still_open',
            'regime': pos['regime']
        })
        cash += final_price * pos['shares']

# Calculate performance metrics
print("\n" + "=" * 80)
print("STRATEGY PERFORMANCE METRICS")
print("=" * 80)

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

initial_capital = 1_000_000
final_capital = equity_df['total'].iloc[-1] if len(equity_df) > 0 else cash

# Basic metrics
total_return = final_capital / initial_capital - 1
n_trades = len(trades_df)

if n_trades > 0:
    # Win/Loss analysis
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winners) / n_trades
    
    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winners['pnl_pct'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_pct'].mean() if len(losers) > 0 else 0
    
    avg_holding = trades_df['holding_days'].mean()
    
    # Risk metrics
    equity_df['returns'] = equity_df['total'].pct_change()
    daily_returns = equity_df['returns'].dropna()
    
    if len(daily_returns) > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        peak = equity_df['total'].cummax()
        drawdown = (peak - equity_df['total']) / peak
        max_drawdown = drawdown.max()
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
    else:
        sharpe = 0
        max_drawdown = 0
        sortino = 0
    
    # By exit reason
    exit_breakdown = trades_df.groupby('exit_reason').agg({
        'ticker': 'count',
        'pnl_pct': 'mean',
        'holding_days': 'mean'
    }).round(4)
    exit_breakdown.columns = ['Count', 'Avg Return', 'Avg Days']
    
    # By regime
    regime_breakdown = trades_df.groupby('regime').agg({
        'ticker': 'count',
        'pnl_pct': 'mean',
        'pnl': 'sum'
    }).round(4)
    regime_breakdown.columns = ['Count', 'Avg Return', 'Total PnL']

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                         STRATEGY PERFORMANCE SUMMARY                           │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  RETURNS                                                                       │
│  ────────────────────────────────────────────────────────────────────────────  │
│  Initial Capital:      {initial_capital:>15,.0f} SAR                                  │
│  Final Capital:        {final_capital:>15,.0f} SAR                                  │
│  Total Return:         {total_return*100:>+14.2f}%                                     │
│  Annualized Return:    {((1+total_return)**(252/len(equity_df))-1)*100 if len(equity_df) > 0 else 0:>+14.2f}%                                     │
│                                                                                │
│  RISK METRICS                                                                  │
│  ────────────────────────────────────────────────────────────────────────────  │
│  Sharpe Ratio:         {sharpe:>15.2f}                                          │
│  Sortino Ratio:        {sortino:>15.2f}                                          │
│  Max Drawdown:         {max_drawdown*100:>14.2f}%                                     │
│                                                                                │
│  TRADE STATISTICS                                                              │
│  ────────────────────────────────────────────────────────────────────────────  │
│  Total Trades:         {n_trades:>15}                                          │
│  Win Rate:             {win_rate*100:>14.1f}%                                     │
│  Profit Factor:        {profit_factor:>15.2f}                                          │
│  Avg Win:              {avg_win*100:>+14.2f}%                                     │
│  Avg Loss:             {avg_loss*100:>+14.2f}%                                     │
│  Avg Holding Days:     {avg_holding:>15.1f}                                          │
│                                                                                │
│  EXPECTANCY                                                                    │
│  ────────────────────────────────────────────────────────────────────────────  │
│  Per Trade Expectancy: {(win_rate*avg_win + (1-win_rate)*avg_loss)*100:>+14.3f}%                                    │
│  Monthly Expectancy:   {(win_rate*avg_win + (1-win_rate)*avg_loss)*100 * (30/avg_holding):>+14.3f}%                                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
""")

print("\nEXIT REASON BREAKDOWN:")
print("-" * 60)
print(exit_breakdown.to_string())

print("\n\nREGIME BREAKDOWN:")
print("-" * 60)
print(regime_breakdown.to_string())

# Top trades
print("\n\nTOP 10 WINNING TRADES:")
print("-" * 60)
top_wins = trades_df.nlargest(10, 'pnl_pct')
for _, t in top_wins.iterrows():
    print(f"  {t['ticker']:<10} {t['entry_date'].strftime('%Y-%m-%d')} -> {t['exit_date'].strftime('%Y-%m-%d')} "
          f"{t['pnl_pct']*100:>+6.1f}% {t['exit_reason']:<15}")

print("\n\nTOP 10 LOSING TRADES:")
print("-" * 60)
top_losses = trades_df.nsmallest(10, 'pnl_pct')
for _, t in top_losses.iterrows():
    print(f"  {t['ticker']:<10} {t['entry_date'].strftime('%Y-%m-%d')} -> {t['exit_date'].strftime('%Y-%m-%d')} "
          f"{t['pnl_pct']*100:>+6.1f}% {t['exit_reason']:<15}")

# Save metrics
metrics = {
    'total_return': total_return,
    'sharpe_ratio': sharpe,
    'sortino_ratio': sortino,
    'max_drawdown': max_drawdown,
    'win_rate': win_rate,
    'profit_factor': profit_factor,
    'avg_win': avg_win,
    'avg_loss': avg_loss,
    'avg_holding_days': avg_holding,
    'total_trades': n_trades,
    'per_trade_expectancy': win_rate*avg_win + (1-win_rate)*avg_loss
}

os.makedirs('output/backtest', exist_ok=True)
with open('output/backtest/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

trades_df.to_csv('output/backtest/trades.csv', index=False)
equity_df.to_csv('output/backtest/equity_curve.csv', index=False)

print(f"\n\nResults saved to output/backtest/")
print("=" * 80)
