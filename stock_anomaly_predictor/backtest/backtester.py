"""
Comprehensive Backtesting Framework

Rigorous backtesting with:
- Walk-forward optimization
- Out-of-sample testing
- Transaction costs and slippage
- Position sizing strategies
- Risk-adjusted performance metrics
- Monte Carlo simulation
- Statistical significance testing
- Drawdown analysis
- Regime-based analysis

Ensures models are validated properly before deployment.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    entry_price: float
    exit_date: datetime = None
    exit_price: float = None
    position_size: float = 1.0
    side: str = 'long'
    signal_strength: float = 1.0
    exit_reason: str = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.side == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price * self.position_size
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * self.position_size
    
    @property
    def holding_period(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    signals_df: pd.DataFrame
    
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_holding_period: float = 0.0
    total_trades: int = 0
    
    # Additional metrics
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0


class Backtester:
    """
    Comprehensive backtesting engine.
    """
    
    def __init__(self,
                 initial_capital: float = None,
                 position_size_pct: float = None,
                 max_positions: int = None,
                 commission: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.001):   # 0.1% slippage
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            max_positions: Maximum concurrent positions
            commission: Commission per trade (as decimal)
            slippage: Slippage per trade (as decimal)
        """
        self.initial_capital = initial_capital or config.backtest.initial_capital
        self.position_size_pct = position_size_pct or config.backtest.position_size_pct
        self.max_positions = max_positions or config.backtest.max_positions
        self.commission = commission
        self.slippage = slippage
        
        # Risk management
        self.stop_loss_pct = config.backtest.stop_loss_pct
        self.take_profit_pct = config.backtest.take_profit_pct
        self.trailing_stop_pct = config.backtest.trailing_stop_pct
        self.max_holding_days = config.backtest.max_holding_days
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        
    def run_backtest(self,
                     price_df: pd.DataFrame,
                     signals_df: pd.DataFrame,
                     signal_column: str = 'buy_signal',
                     price_column: str = 'close') -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            price_df: DataFrame with price data
            signals_df: DataFrame with signals
            signal_column: Column name for entry signals
            price_column: Column name for prices
            
        Returns:
            BacktestResult with performance metrics
        """
        print("=" * 60)
        print("Running Backtest")
        print("=" * 60)
        
        # Align data
        df = price_df.copy()
        for col in signals_df.columns:
            if col not in df.columns:
                df[col] = signals_df[col]
        
        # Initialize
        capital = self.initial_capital
        position = 0
        open_trades: List[Trade] = []
        all_trades: List[Trade] = []
        equity = [capital]
        
        # Track highest price for trailing stop
        highest_price_since_entry = {}
        
        for i in range(1, len(df)):
            date = df.index[i]
            price = df[price_column].iloc[i]
            prev_price = df[price_column].iloc[i-1]
            
            # Check exit conditions for open trades
            trades_to_close = []
            
            for trade in open_trades:
                trade_idx = open_trades.index(trade)
                
                # Update highest price for trailing stop
                if trade_idx not in highest_price_since_entry:
                    highest_price_since_entry[trade_idx] = trade.entry_price
                highest_price_since_entry[trade_idx] = max(
                    highest_price_since_entry[trade_idx], 
                    price
                )
                
                exit_reason = None
                
                # Stop loss
                if price <= trade.entry_price * (1 - self.stop_loss_pct):
                    exit_reason = 'stop_loss'
                
                # Take profit
                elif price >= trade.entry_price * (1 + self.take_profit_pct):
                    exit_reason = 'take_profit'
                
                # Trailing stop
                elif price <= highest_price_since_entry[trade_idx] * (1 - self.trailing_stop_pct):
                    exit_reason = 'trailing_stop'
                
                # Max holding period
                elif trade.holding_period >= self.max_holding_days:
                    exit_reason = 'max_holding'
                
                # Exit signal (if we add exit signals)
                # elif signals_df.get('exit_signal', pd.Series(False)).iloc[i]:
                #     exit_reason = 'exit_signal'
                
                if exit_reason:
                    trades_to_close.append((trade, exit_reason))
            
            # Close trades
            for trade, reason in trades_to_close:
                # Apply slippage to exit
                exit_price = price * (1 - self.slippage)
                
                trade.exit_date = date
                trade.exit_price = exit_price
                trade.exit_reason = reason
                
                # Calculate P&L
                pnl = trade.pnl * capital * trade.position_size
                pnl -= capital * trade.position_size * self.commission  # Exit commission
                
                capital += pnl
                position -= trade.position_size
                
                open_trades.remove(trade)
                all_trades.append(trade)
            
            # Check entry signals
            if signal_column in df.columns:
                signal = df[signal_column].iloc[i]
                
                if signal and len(open_trades) < self.max_positions:
                    # Position sizing
                    signal_strength = df.get('signal_strength', pd.Series(1.0)).iloc[i]
                    if pd.isna(signal_strength):
                        signal_strength = 1.0
                    
                    pos_size = self.position_size_pct * signal_strength
                    pos_size = min(pos_size, 1 - position)  # Don't exceed 100%
                    
                    if pos_size > 0.01:  # Minimum position size
                        # Apply slippage to entry
                        entry_price = price * (1 + self.slippage)
                        
                        trade = Trade(
                            entry_date=date,
                            entry_price=entry_price,
                            position_size=pos_size,
                            signal_strength=signal_strength
                        )
                        
                        # Entry commission
                        capital -= capital * pos_size * self.commission
                        position += pos_size
                        
                        open_trades.append(trade)
            
            # Update equity
            # Mark-to-market for open positions
            mtm_pnl = sum(
                (price - t.entry_price) / t.entry_price * capital * t.position_size
                for t in open_trades
            )
            equity.append(capital + mtm_pnl)
        
        # Close any remaining open trades at end
        for trade in open_trades:
            trade.exit_date = df.index[-1]
            trade.exit_price = df[price_column].iloc[-1]
            trade.exit_reason = 'end_of_backtest'
            all_trades.append(trade)
        
        # Create equity curve
        equity_curve = pd.Series(equity, index=df.index[:len(equity)])
        daily_returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        result = self._calculate_metrics(all_trades, equity_curve, daily_returns, df)
        
        return result
    
    def _calculate_metrics(self,
                           trades: List[Trade],
                           equity_curve: pd.Series,
                           daily_returns: pd.Series,
                           signals_df: pd.DataFrame) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        """
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            signals_df=signals_df
        )
        
        result.total_trades = len(trades)
        
        if result.total_trades == 0:
            print("  No trades executed")
            return result
        
        # Trade-based metrics
        trade_returns = [t.pnl for t in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        result.win_rate = len(winning_trades) / len(trades) if trades else 0
        result.avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        result.avg_holding_period = np.mean([t.holding_period for t in trades])
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Return metrics
        result.total_return = (equity_curve.iloc[-1] / self.initial_capital - 1)
        
        trading_days = len(daily_returns)
        years = trading_days / 252
        result.annual_return = (1 + result.total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        result.volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 4%)
        risk_free_daily = 0.04 / 252
        excess_returns = daily_returns - risk_free_daily
        result.sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0 else 0
        )
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        result.sortino_ratio = (
            (result.annual_return - 0.04) / downside_std
            if downside_std > 0 else 0
        )
        
        # Drawdown analysis
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        result.calmar_ratio = (
            result.annual_return / result.max_drawdown
            if result.max_drawdown > 0 else 0
        )
        
        # Recovery factor
        result.recovery_factor = (
            result.total_return / result.max_drawdown
            if result.max_drawdown > 0 else 0
        )
        
        # Distribution metrics
        result.skewness = stats.skew(daily_returns)
        result.kurtosis = stats.kurtosis(daily_returns)
        
        # VaR and CVaR
        result.var_95 = np.percentile(daily_returns, 5)
        result.cvar_95 = daily_returns[daily_returns <= result.var_95].mean()
        
        # Print summary
        print(f"\nBacktest Results Summary:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Total Return: {result.total_return:.1%}")
        print(f"  Annual Return: {result.annual_return:.1%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.1%}")
        print(f"  Calmar Ratio: {result.calmar_ratio:.2f}")
        print(f"  Avg Holding Period: {result.avg_holding_period:.1f} days")
        
        return result
    
    def walk_forward_optimization(self,
                                  price_df: pd.DataFrame,
                                  signals_df: pd.DataFrame,
                                  n_splits: int = None,
                                  train_ratio: float = None) -> List[BacktestResult]:
        """
        Walk-forward optimization for robust out-of-sample testing.
        
        Args:
            price_df: Price DataFrame
            signals_df: Signals DataFrame
            n_splits: Number of walk-forward windows
            train_ratio: Training period ratio
            
        Returns:
            List of BacktestResults for each window
        """
        n_splits = n_splits or config.backtest.walk_forward_windows
        train_ratio = train_ratio or config.backtest.train_ratio
        
        print("=" * 60)
        print(f"Running Walk-Forward Analysis ({n_splits} windows)")
        print("=" * 60)
        
        n = len(price_df)
        window_size = n // n_splits
        
        results = []
        
        for i in range(n_splits):
            start_idx = i * window_size
            end_idx = min((i + 2) * window_size, n)  # Overlapping windows
            
            window_price = price_df.iloc[start_idx:end_idx]
            window_signals = signals_df.iloc[start_idx:end_idx]
            
            # Split into train/test
            train_end = int(len(window_price) * train_ratio)
            
            test_price = window_price.iloc[train_end:]
            test_signals = window_signals.iloc[train_end:]
            
            print(f"\nWindow {i+1}/{n_splits}:")
            print(f"  Test period: {test_price.index[0].date()} to {test_price.index[-1].date()}")
            
            # Run backtest on test period
            result = self.run_backtest(test_price, test_signals)
            results.append(result)
        
        # Aggregate results
        self._print_walk_forward_summary(results)
        
        return results
    
    def _print_walk_forward_summary(self, results: List[BacktestResult]):
        """Print summary of walk-forward results."""
        print("\n" + "=" * 60)
        print("Walk-Forward Summary")
        print("=" * 60)
        
        sharpes = [r.sharpe_ratio for r in results if r.total_trades > 0]
        returns = [r.total_return for r in results if r.total_trades > 0]
        win_rates = [r.win_rate for r in results if r.total_trades > 0]
        
        if sharpes:
            print(f"Average Sharpe: {np.mean(sharpes):.2f} (std: {np.std(sharpes):.2f})")
            print(f"Average Return: {np.mean(returns):.1%} (std: {np.std(returns):.1%})")
            print(f"Average Win Rate: {np.mean(win_rates):.1%}")
            print(f"Positive Windows: {sum(1 for r in returns if r > 0)}/{len(returns)}")
        else:
            print("No valid windows with trades")
    
    def monte_carlo_simulation(self,
                               result: BacktestResult,
                               n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation for robustness analysis.
        
        Shuffles trade order to assess luck vs skill.
        
        Args:
            result: BacktestResult to simulate
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with simulation statistics
        """
        print(f"\nRunning Monte Carlo Simulation ({n_simulations} iterations)...")
        
        trade_returns = [t.pnl for t in result.trades]
        
        if len(trade_returns) < 10:
            print("  Insufficient trades for Monte Carlo")
            return {}
        
        simulated_returns = []
        simulated_sharpes = []
        simulated_max_dd = []
        
        for _ in range(n_simulations):
            # Shuffle trade returns
            shuffled = np.random.permutation(trade_returns)
            
            # Calculate equity curve
            equity = [self.initial_capital]
            for ret in shuffled:
                equity.append(equity[-1] * (1 + ret))
            
            equity = pd.Series(equity)
            
            # Metrics
            total_ret = equity.iloc[-1] / self.initial_capital - 1
            daily_rets = equity.pct_change().dropna()
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
            
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max
            max_dd = abs(drawdown.min())
            
            simulated_returns.append(total_ret)
            simulated_sharpes.append(sharpe)
            simulated_max_dd.append(max_dd)
        
        mc_results = {
            'return_mean': np.mean(simulated_returns),
            'return_std': np.std(simulated_returns),
            'return_5th': np.percentile(simulated_returns, 5),
            'return_95th': np.percentile(simulated_returns, 95),
            'sharpe_mean': np.mean(simulated_sharpes),
            'sharpe_std': np.std(simulated_sharpes),
            'max_dd_mean': np.mean(simulated_max_dd),
            'max_dd_95th': np.percentile(simulated_max_dd, 95),
            'actual_return_percentile': stats.percentileofscore(simulated_returns, result.total_return),
            'actual_sharpe_percentile': stats.percentileofscore(simulated_sharpes, result.sharpe_ratio)
        }
        
        print(f"  Actual return percentile: {mc_results['actual_return_percentile']:.1f}%")
        print(f"  Actual Sharpe percentile: {mc_results['actual_sharpe_percentile']:.1f}%")
        
        return mc_results
    
    def benchmark_comparison(self,
                             result: BacktestResult,
                             benchmark_returns: pd.Series) -> Dict:
        """
        Compare strategy to benchmark (buy-and-hold).
        
        Args:
            result: Backtest result
            benchmark_returns: Benchmark return series
            
        Returns:
            Comparison metrics
        """
        print("\nBenchmark Comparison (vs Buy-and-Hold):")
        
        # Align
        benchmark_aligned = benchmark_returns.reindex(result.daily_returns.index)
        
        # Benchmark metrics
        benchmark_total = (1 + benchmark_aligned).prod() - 1
        benchmark_vol = benchmark_aligned.std() * np.sqrt(252)
        benchmark_sharpe = benchmark_aligned.mean() / benchmark_aligned.std() * np.sqrt(252)
        
        # Alpha and Beta
        if len(result.daily_returns) > 30:
            cov = result.daily_returns.cov(benchmark_aligned)
            var = benchmark_aligned.var()
            beta = cov / var if var > 0 else 1
            
            strategy_annual = result.annual_return
            benchmark_annual = (1 + benchmark_total) ** (252/len(benchmark_aligned)) - 1
            alpha = strategy_annual - beta * benchmark_annual
        else:
            alpha = 0
            beta = 1
        
        comparison = {
            'strategy_return': result.total_return,
            'benchmark_return': benchmark_total,
            'excess_return': result.total_return - benchmark_total,
            'strategy_sharpe': result.sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': (result.annual_return - benchmark_annual) / (result.daily_returns - benchmark_aligned).std() / np.sqrt(252) if (result.daily_returns - benchmark_aligned).std() > 0 else 0
        }
        
        print(f"  Strategy Return: {comparison['strategy_return']:.1%}")
        print(f"  Benchmark Return: {comparison['benchmark_return']:.1%}")
        print(f"  Excess Return: {comparison['excess_return']:.1%}")
        print(f"  Alpha (annual): {comparison['alpha']:.1%}")
        print(f"  Beta: {comparison['beta']:.2f}")
        
        return comparison


class SignalAnalyzer:
    """
    Analyze signal quality and predictive power.
    """
    
    def __init__(self, signals_df: pd.DataFrame, returns: pd.Series):
        """
        Initialize signal analyzer.
        
        Args:
            signals_df: DataFrame with signals
            returns: Return series
        """
        self.signals_df = signals_df
        self.returns = returns
    
    def analyze_signal_hit_rate(self,
                                signal_column: str,
                                forward_days: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        """
        Analyze hit rate of signals at various forward horizons.
        
        Args:
            signal_column: Signal column to analyze
            forward_days: List of forward periods to analyze
            
        Returns:
            Dictionary with hit rates
        """
        results = {}
        
        signal = self.signals_df[signal_column]
        signal_dates = signal[signal == True].index
        
        for days in forward_days:
            forward_returns = self.returns.shift(-days).rolling(days).sum()
            
            hits = 0
            total = 0
            
            for date in signal_dates:
                if date in forward_returns.index:
                    fwd_ret = forward_returns.loc[date]
                    if not pd.isna(fwd_ret):
                        total += 1
                        if fwd_ret > 0:
                            hits += 1
            
            hit_rate = hits / total if total > 0 else 0
            results[f'{days}d_hit_rate'] = hit_rate
            results[f'{days}d_signals'] = total
        
        return results
    
    def analyze_signal_magnitude(self,
                                 signal_column: str,
                                 forward_days: int = 5) -> Dict:
        """
        Analyze average return magnitude when signal fires.
        
        Args:
            signal_column: Signal column
            forward_days: Forward period
            
        Returns:
            Dictionary with magnitude analysis
        """
        signal = self.signals_df[signal_column]
        signal_dates = signal[signal == True].index
        non_signal_dates = signal[signal == False].index
        
        forward_returns = self.returns.shift(-forward_days).rolling(forward_days).sum()
        
        signal_returns = forward_returns.loc[signal_dates].dropna()
        non_signal_returns = forward_returns.loc[non_signal_dates].dropna()
        
        results = {
            'avg_signal_return': signal_returns.mean() if len(signal_returns) > 0 else 0,
            'avg_non_signal_return': non_signal_returns.mean() if len(non_signal_returns) > 0 else 0,
            'signal_std': signal_returns.std() if len(signal_returns) > 0 else 0,
            'lift': (signal_returns.mean() - non_signal_returns.mean()) / non_signal_returns.std() if non_signal_returns.std() > 0 else 0
        }
        
        # T-test for statistical significance
        if len(signal_returns) > 5 and len(non_signal_returns) > 5:
            t_stat, p_value = stats.ttest_ind(signal_returns, non_signal_returns)
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            results['statistically_significant'] = p_value < 0.05
        
        return results


if __name__ == "__main__":
    # Test backtester with dummy data
    print("Testing Backtester...")
    
    # Create dummy data
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    price_df = pd.DataFrame({
        'close': 10 * np.exp(np.cumsum(np.random.randn(500) * 0.02)),
        'returns': np.random.randn(500) * 0.02
    }, index=dates)
    
    # Create dummy signals (random with ~5% signal rate)
    signals_df = pd.DataFrame({
        'buy_signal': np.random.random(500) < 0.05,
        'signal_strength': np.random.random(500) * 0.5 + 0.5
    }, index=dates)
    
    # Run backtest
    backtester = Backtester()
    result = backtester.run_backtest(price_df, signals_df)
    
    # Monte Carlo
    mc_results = backtester.monte_carlo_simulation(result)
    
    # Benchmark comparison
    benchmark_comparison = backtester.benchmark_comparison(result, price_df['returns'])
    
    print("\nBacktest complete!")
