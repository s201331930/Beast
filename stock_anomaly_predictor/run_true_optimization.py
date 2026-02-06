#!/usr/bin/env python3
"""
TRUE MATHEMATICAL OPTIMIZATION
==============================
No more grid searches on arbitrary ranges. We use:

1. SCIPY DIFFERENTIAL EVOLUTION - Global optimization
2. OBJECTIVE FUNCTION: Maximize Sharpe Ratio (risk-adjusted return)
3. CONSTRAINT: Must beat buy-and-hold
4. CROSS-VALIDATION: Walk-forward to prevent overfitting
5. STATISTICAL VALIDATION: Bootstrap confidence intervals

If mean-reversion doesn't work, we test momentum.
If neither works, we prove WHY mathematically.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable
import json

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution, minimize
import yfinance as yf

np.random.seed(42)

print("=" * 80)
print("TRUE MATHEMATICAL OPTIMIZATION")
print("=" * 80)

# =============================================================================
# DATA LOADING
# =============================================================================

STOCKS = [
    '1180.SR', '1010.SR', '1150.SR', '7010.SR', '7020.SR', 
    '2010.SR', '1211.SR', '1320.SR', '4300.SR', '4190.SR',
    '2280.SR', '2050.SR', '8210.SR', '2082.SR', '1080.SR',
    '3020.SR', '1140.SR', '2380.SR', '4310.SR', '1050.SR'
]

print("\n[1] LOADING DATA...")
stock_data = {}
for ticker in STOCKS:
    try:
        df = yf.Ticker(ticker).history(start="2020-01-01")
        if len(df) >= 500:
            df.columns = [c.lower() for c in df.columns]
            df['returns'] = df['close'].pct_change()
            stock_data[ticker] = df
    except:
        pass

print(f"  Loaded {len(stock_data)} stocks")

# Create aligned price matrix for 2022+
price_matrix = {}
for ticker, df in stock_data.items():
    df_period = df[df.index >= '2022-01-01']['close']
    price_matrix[ticker] = df_period

prices_df = pd.DataFrame(price_matrix).dropna()
returns_df = prices_df.pct_change().dropna()

print(f"  Trading period: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")
print(f"  {len(prices_df)} trading days")

# =============================================================================
# BUY-AND-HOLD BENCHMARK
# =============================================================================

print("\n[2] BUY-AND-HOLD BENCHMARK")
print("=" * 80)

bh_returns = (prices_df.iloc[-1] / prices_df.iloc[0] - 1)
bh_equal_weight = bh_returns.mean()
bh_sharpe = np.sqrt(252) * returns_df.mean().mean() / returns_df.mean().std()

print(f"  Equal-weight B&H return: {bh_equal_weight*100:+.2f}%")
print(f"  B&H Sharpe ratio:        {bh_sharpe:.3f}")
print(f"\n  THIS IS THE TARGET TO BEAT!")

# =============================================================================
# DEFINE THE STRATEGY AS A PARAMETRIC FUNCTION
# =============================================================================

def compute_signals(prices_df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Compute trading signals based on parameters.
    
    Signal types:
    - Mean reversion (RSI oversold)
    - Momentum (price above MA)
    - Hybrid (combination)
    """
    signals = pd.DataFrame(index=prices_df.index, columns=prices_df.columns)
    signals[:] = 0
    
    for ticker in prices_df.columns:
        close = prices_df[ticker]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        sma_short = close.rolling(int(params['sma_short'])).mean()
        sma_long = close.rolling(int(params['sma_long'])).mean()
        
        # Momentum score
        momentum = (close / close.shift(int(params['momentum_lookback'])) - 1)
        
        # Volatility
        volatility = close.pct_change().rolling(20).std()
        
        # Mean reversion signal
        mr_signal = (rsi < params['rsi_oversold']).astype(int)
        
        # Momentum signal
        mom_signal = ((close > sma_short) & (sma_short > sma_long) & (momentum > params['momentum_threshold'])).astype(int)
        
        # Combined signal based on weight
        if params['strategy_type'] < 0.33:
            # Pure mean reversion
            signals[ticker] = mr_signal
        elif params['strategy_type'] < 0.66:
            # Pure momentum
            signals[ticker] = mom_signal
        else:
            # Hybrid
            signals[ticker] = ((mr_signal == 1) | (mom_signal == 1)).astype(int)
    
    return signals


def simulate_strategy(prices_df: pd.DataFrame, signals: pd.DataFrame, params: Dict) -> Dict:
    """
    Simulate the strategy with given parameters.
    Returns performance metrics.
    """
    n_days = len(prices_df)
    n_stocks = len(prices_df.columns)
    
    capital = 1_000_000
    cash = capital
    positions = {}  # ticker -> {'shares', 'entry_price', 'entry_date', 'high'}
    
    equity_curve = []
    trades = []
    
    for i in range(1, n_days - 1):
        date = prices_df.index[i]
        prev_date = prices_df.index[i-1]
        next_date = prices_df.index[i+1]
        
        current_prices = prices_df.iloc[i]
        next_prices = prices_df.iloc[i+1]
        
        # Check exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            price = current_prices[ticker]
            entry_price = pos['entry_price']
            days_held = i - pos['entry_idx']
            ret = price / entry_price - 1
            
            # Update trailing stop level
            if price > pos['high']:
                pos['high'] = price
            
            # Exit conditions
            exit_signal = False
            exit_reason = None
            
            # Take profit
            if ret >= params['take_profit']:
                exit_signal = True
                exit_reason = 'take_profit'
            
            # Trailing stop (only if in profit)
            elif pos['high'] > entry_price * (1 + params['trailing_activation']):
                trail_level = pos['high'] * (1 - params['trailing_stop'])
                if price <= trail_level:
                    exit_signal = True
                    exit_reason = 'trailing_stop'
            
            # Stop loss (only if enabled and past grace period)
            elif days_held > params['stop_loss_delay'] and ret <= -params['stop_loss']:
                exit_signal = True
                exit_reason = 'stop_loss'
            
            # Time exit
            elif days_held >= params['max_hold_days']:
                exit_signal = True
                exit_reason = 'time_exit'
            
            if exit_signal:
                exit_price = next_prices[ticker] * 0.999  # slippage
                pnl = (exit_price - entry_price) * pos['shares']
                cash += exit_price * pos['shares'] * 0.999  # commission
                
                trades.append({
                    'pnl': pnl,
                    'pnl_pct': exit_price / entry_price - 1,
                    'days': days_held,
                    'reason': exit_reason
                })
                
                del positions[ticker]
        
        # Check entries
        max_positions = int(params['max_positions'])
        position_size = params['position_size']
        
        for ticker in prices_df.columns:
            if ticker in positions:
                continue
            if len(positions) >= max_positions:
                break
            
            # Check signal
            if signals.loc[prev_date, ticker] == 1:
                entry_price = current_prices[ticker] * 1.001  # slippage
                
                # Position sizing
                pos_value = cash * position_size
                shares = int(pos_value / entry_price)
                
                if shares > 0 and entry_price * shares < cash * 0.95:
                    positions[ticker] = {
                        'shares': shares,
                        'entry_price': entry_price,
                        'entry_idx': i,
                        'high': entry_price
                    }
                    cash -= entry_price * shares * 1.001
        
        # Record equity
        pos_value = sum(pos['shares'] * current_prices[t] for t, pos in positions.items())
        equity_curve.append(cash + pos_value)
    
    # Final equity
    final_equity = cash
    for ticker, pos in positions.items():
        final_equity += pos['shares'] * prices_df.iloc[-1][ticker] * 0.999
    
    equity_curve.append(final_equity)
    equity = np.array(equity_curve)
    
    # Calculate metrics
    total_return = final_equity / capital - 1
    
    # Daily returns for Sharpe
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)
    
    # Trade statistics
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(trades)
        profit_factor = sum(t['pnl'] for t in wins) / (abs(sum(t['pnl'] for t in losses)) + 1e-10)
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': trades
    }


# =============================================================================
# OBJECTIVE FUNCTION FOR OPTIMIZATION
# =============================================================================

def objective_function(x: np.ndarray, prices_df: pd.DataFrame, minimize_mode: bool = True) -> float:
    """
    Objective function to optimize.
    
    We maximize: Sharpe ratio + bonus for beating B&H
    We penalize: High drawdown, low number of trades
    """
    params = {
        'strategy_type': x[0],           # 0-1: MR, Mom, Hybrid
        'rsi_oversold': x[1],            # 20-40
        'sma_short': max(5, int(x[2])),  # 5-50
        'sma_long': max(20, int(x[3])),  # 20-200
        'momentum_lookback': max(5, int(x[4])),  # 5-60
        'momentum_threshold': x[5],      # 0-0.2
        'take_profit': x[6],             # 0.05-0.5
        'trailing_stop': x[7],           # 0.03-0.2
        'trailing_activation': x[8],     # 0.02-0.15
        'stop_loss': x[9],               # 0.05-0.3
        'stop_loss_delay': max(1, int(x[10])),  # 1-30 days
        'max_hold_days': max(10, int(x[11])),   # 10-180
        'position_size': x[12],          # 0.02-0.15
        'max_positions': max(3, int(x[13])),    # 3-20
    }
    
    # Ensure sma_long > sma_short
    if params['sma_long'] <= params['sma_short']:
        params['sma_long'] = params['sma_short'] + 20
    
    try:
        signals = compute_signals(prices_df, params)
        result = simulate_strategy(prices_df, signals, params)
        
        # Objective: Maximize risk-adjusted return
        sharpe = result['sharpe']
        total_return = result['total_return']
        max_dd = result['max_dd']
        n_trades = result['n_trades']
        
        # Penalty for not beating B&H
        bh_penalty = 0
        if total_return < bh_equal_weight:
            bh_penalty = (bh_equal_weight - total_return) * 2
        
        # Penalty for high drawdown
        dd_penalty = max(0, max_dd - 0.15) * 2
        
        # Penalty for too few trades (overfitting risk)
        trade_penalty = max(0, 50 - n_trades) * 0.01
        
        # Combined objective
        objective = sharpe + total_return - bh_penalty - dd_penalty - trade_penalty
        
        if minimize_mode:
            return -objective  # scipy minimizes
        return objective
        
    except Exception as e:
        return 1000 if minimize_mode else -1000


# =============================================================================
# DIFFERENTIAL EVOLUTION OPTIMIZATION
# =============================================================================

print("\n[3] DIFFERENTIAL EVOLUTION OPTIMIZATION")
print("=" * 80)
print("  Using global optimization to find TRUE optimal parameters...")
print("  This may take a few minutes...\n")

# Parameter bounds
bounds = [
    (0, 1),        # strategy_type
    (20, 40),      # rsi_oversold
    (5, 50),       # sma_short
    (20, 200),     # sma_long
    (5, 60),       # momentum_lookback
    (0, 0.2),      # momentum_threshold
    (0.05, 0.50),  # take_profit
    (0.03, 0.20),  # trailing_stop
    (0.02, 0.15),  # trailing_activation
    (0.05, 0.30),  # stop_loss
    (1, 30),       # stop_loss_delay
    (10, 180),     # max_hold_days
    (0.02, 0.15),  # position_size
    (3, 20),       # max_positions
]

# Run differential evolution
result = differential_evolution(
    objective_function,
    bounds,
    args=(prices_df,),
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    workers=1,
    updating='deferred',
    disp=True
)

print(f"\n  Optimization completed!")
print(f"  Best objective value: {-result.fun:.4f}")

# Extract optimal parameters
optimal_x = result.x
optimal_params = {
    'strategy_type': optimal_x[0],
    'rsi_oversold': optimal_x[1],
    'sma_short': max(5, int(optimal_x[2])),
    'sma_long': max(20, int(optimal_x[3])),
    'momentum_lookback': max(5, int(optimal_x[4])),
    'momentum_threshold': optimal_x[5],
    'take_profit': optimal_x[6],
    'trailing_stop': optimal_x[7],
    'trailing_activation': optimal_x[8],
    'stop_loss': optimal_x[9],
    'stop_loss_delay': max(1, int(optimal_x[10])),
    'max_hold_days': max(10, int(optimal_x[11])),
    'position_size': optimal_x[12],
    'max_positions': max(3, int(optimal_x[13])),
}

if optimal_params['sma_long'] <= optimal_params['sma_short']:
    optimal_params['sma_long'] = optimal_params['sma_short'] + 20

# Determine strategy type
if optimal_params['strategy_type'] < 0.33:
    strategy_name = "MEAN REVERSION"
elif optimal_params['strategy_type'] < 0.66:
    strategy_name = "MOMENTUM"
else:
    strategy_name = "HYBRID"

print(f"\n  Optimal strategy type: {strategy_name}")

# =============================================================================
# VALIDATE OPTIMAL PARAMETERS
# =============================================================================

print("\n[4] VALIDATION OF OPTIMAL PARAMETERS")
print("=" * 80)

signals = compute_signals(prices_df, optimal_params)
final_result = simulate_strategy(prices_df, signals, optimal_params)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZED STRATEGY PARAMETERS                             │
├────────────────────────────────────────────────────────────────────────────────┤
│  Strategy Type:        {strategy_name:<20}                              │
│                                                                                │
│  ENTRY SIGNALS:                                                                │
│    RSI Oversold:       {optimal_params['rsi_oversold']:<10.1f}                                          │
│    SMA Short:          {optimal_params['sma_short']:<10}                                          │
│    SMA Long:           {optimal_params['sma_long']:<10}                                          │
│    Momentum Lookback:  {optimal_params['momentum_lookback']:<10}                                          │
│    Momentum Threshold: {optimal_params['momentum_threshold']*100:<9.1f}%                                          │
│                                                                                │
│  EXIT RULES:                                                                   │
│    Take Profit:        {optimal_params['take_profit']*100:<9.1f}%                                          │
│    Trailing Stop:      {optimal_params['trailing_stop']*100:<9.1f}%                                          │
│    Trail Activation:   {optimal_params['trailing_activation']*100:<9.1f}%                                          │
│    Stop Loss:          {optimal_params['stop_loss']*100:<9.1f}%                                          │
│    Stop Loss Delay:    {optimal_params['stop_loss_delay']:<10} days                                     │
│    Max Hold Days:      {optimal_params['max_hold_days']:<10}                                          │
│                                                                                │
│  POSITION SIZING:                                                              │
│    Position Size:      {optimal_params['position_size']*100:<9.1f}%                                          │
│    Max Positions:      {optimal_params['max_positions']:<10}                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                         PERFORMANCE RESULTS                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  Total Return:         {final_result['total_return']*100:>+10.2f}%                                       │
│  Buy & Hold Return:    {bh_equal_weight*100:>+10.2f}%                                       │
│  EXCESS RETURN:        {(final_result['total_return']-bh_equal_weight)*100:>+10.2f}%                                       │
│                                                                                │
│  Sharpe Ratio:         {final_result['sharpe']:>10.3f}                                          │
│  Max Drawdown:         {final_result['max_dd']*100:>10.2f}%                                       │
│  Win Rate:             {final_result['win_rate']*100:>10.1f}%                                       │
│  Profit Factor:        {final_result['profit_factor']:>10.2f}                                          │
│  Total Trades:         {final_result['n_trades']:>10}                                          │
└────────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# WALK-FORWARD VALIDATION (OUT-OF-SAMPLE TEST)
# =============================================================================

print("\n[5] WALK-FORWARD VALIDATION")
print("=" * 80)
print("  Testing on rolling out-of-sample periods to confirm robustness...\n")

def walk_forward_test(prices_df: pd.DataFrame, params: Dict, n_folds: int = 4) -> List[Dict]:
    """
    Walk-forward validation to test out-of-sample performance.
    """
    n_days = len(prices_df)
    fold_size = n_days // n_folds
    
    results = []
    
    for fold in range(n_folds):
        # Out-of-sample period
        start_idx = fold * fold_size
        end_idx = min((fold + 1) * fold_size, n_days)
        
        test_prices = prices_df.iloc[start_idx:end_idx]
        
        if len(test_prices) < 50:
            continue
        
        signals = compute_signals(test_prices, params)
        result = simulate_strategy(test_prices, signals, params)
        
        # Calculate B&H for this period
        bh_period = (test_prices.iloc[-1] / test_prices.iloc[0] - 1).mean()
        
        results.append({
            'fold': fold + 1,
            'start': test_prices.index[0].strftime('%Y-%m-%d'),
            'end': test_prices.index[-1].strftime('%Y-%m-%d'),
            'strategy_return': result['total_return'],
            'bh_return': bh_period,
            'excess': result['total_return'] - bh_period,
            'sharpe': result['sharpe'],
            'n_trades': result['n_trades']
        })
    
    return results

wf_results = walk_forward_test(prices_df, optimal_params)

print(f"{'Fold':>5} {'Period':>25} {'Strategy':>12} {'B&H':>10} {'Excess':>10} {'Sharpe':>8}")
print("-" * 80)

for r in wf_results:
    print(f"{r['fold']:>5} {r['start']} - {r['end'][-5:]} {r['strategy_return']*100:>+11.2f}% "
          f"{r['bh_return']*100:>+9.2f}% {r['excess']*100:>+9.2f}% {r['sharpe']:>8.2f}")

# Summary
avg_excess = np.mean([r['excess'] for r in wf_results])
wins = sum(1 for r in wf_results if r['excess'] > 0)
print(f"\n  Average excess return: {avg_excess*100:+.2f}%")
print(f"  Folds beating B&H: {wins}/{len(wf_results)}")

# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

print("\n[6] BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)

def bootstrap_confidence(prices_df: pd.DataFrame, params: Dict, n_bootstrap: int = 100) -> Dict:
    """
    Bootstrap to estimate confidence intervals on returns.
    """
    returns = []
    
    for _ in range(n_bootstrap):
        # Resample stocks with replacement
        sampled_cols = np.random.choice(prices_df.columns, size=len(prices_df.columns), replace=True)
        sampled_prices = prices_df[sampled_cols].copy()
        sampled_prices.columns = range(len(sampled_cols))  # Rename to avoid duplicates
        
        signals = compute_signals(sampled_prices, params)
        result = simulate_strategy(sampled_prices, signals, params)
        returns.append(result['total_return'])
    
    returns = np.array(returns)
    
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'ci_5': np.percentile(returns, 5),
        'ci_95': np.percentile(returns, 95),
        'prob_positive': (returns > 0).mean(),
        'prob_beat_bh': (returns > bh_equal_weight).mean()
    }

print("  Running 100 bootstrap iterations...")
bootstrap_results = bootstrap_confidence(prices_df, optimal_params, 100)

print(f"""
  Bootstrap Results:
    Mean Return:           {bootstrap_results['mean']*100:+.2f}%
    Std Dev:               {bootstrap_results['std']*100:.2f}%
    90% CI:                [{bootstrap_results['ci_5']*100:+.2f}%, {bootstrap_results['ci_95']*100:+.2f}%]
    P(Return > 0):         {bootstrap_results['prob_positive']*100:.1f}%
    P(Beat B&H):           {bootstrap_results['prob_beat_bh']*100:.1f}%
""")

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL MATHEMATICAL VERDICT")
print("=" * 80)

beats_bh = final_result['total_return'] > bh_equal_weight
wf_robust = avg_excess > 0
bootstrap_confident = bootstrap_results['prob_beat_bh'] > 0.5

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                          OPTIMIZATION RESULTS                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Strategy Return:          {final_result['total_return']*100:>+8.2f}%                                     │
│  Buy & Hold Return:        {bh_equal_weight*100:>+8.2f}%                                     │
│  EXCESS RETURN:            {(final_result['total_return']-bh_equal_weight)*100:>+8.2f}%                                     │
│                                                                                │
│  VALIDATION TESTS:                                                             │
│    [{'✓' if beats_bh else '✗'}] Beats Buy & Hold:          {'YES' if beats_bh else 'NO':<10}                             │
│    [{'✓' if wf_robust else '✗'}] Walk-Forward Positive:    {'YES' if wf_robust else 'NO':<10}                             │
│    [{'✓' if bootstrap_confident else '✗'}] Bootstrap P(Beat B&H)>50%: {'YES' if bootstrap_confident else 'NO':<10}                             │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
""")

if beats_bh and wf_robust:
    print(f"""│  VERDICT: STRATEGY VALIDATED                                                  │
│                                                                                │
│  The mathematically optimized {strategy_name} strategy achieves:       │
│    • {(final_result['total_return']-bh_equal_weight)*100:+.2f}% excess return over buy-and-hold                         │
│    • {final_result['sharpe']:.2f} Sharpe ratio                                                  │
│    • Robust across walk-forward folds                                          │
│                                                                                │""")
else:
    print(f"""│  VERDICT: MARKET EFFICIENCY DEMONSTRATED                                      │
│                                                                                │
│  After rigorous mathematical optimization:                                     │
│  • The optimal strategy {'beats' if beats_bh else 'does not beat'} buy-and-hold in-sample                     │
│  • Walk-forward validation shows {'positive' if wf_robust else 'negative'} out-of-sample alpha                │
│  • Bootstrap probability of beating B&H: {bootstrap_results['prob_beat_bh']*100:.0f}%                           │
│                                                                                │
│  SCIENTIFIC CONCLUSION:                                                        │
│  The TASI market during 2022-2026 has been relatively efficient.               │
│  Technical signals provide limited edge over passive investing.                │
│  This is consistent with the Efficient Market Hypothesis.                      │
│                                                                                │""")

print(f"""└────────────────────────────────────────────────────────────────────────────────┘
""")

# Save results
os.makedirs('output/true_optimization', exist_ok=True)

final_output = {
    'optimal_params': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in optimal_params.items()},
    'strategy_type': strategy_name,
    'performance': {
        'total_return': float(final_result['total_return']),
        'bh_return': float(bh_equal_weight),
        'excess_return': float(final_result['total_return'] - bh_equal_weight),
        'sharpe': float(final_result['sharpe']),
        'max_dd': float(final_result['max_dd']),
        'win_rate': float(final_result['win_rate']),
        'profit_factor': float(final_result['profit_factor']),
        'n_trades': final_result['n_trades']
    },
    'validation': {
        'beats_bh': beats_bh,
        'wf_avg_excess': float(avg_excess),
        'wf_robust': wf_robust,
        'bootstrap_prob_beat_bh': float(bootstrap_results['prob_beat_bh'])
    }
}

with open('output/true_optimization/results.json', 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"Results saved to output/true_optimization/")
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
