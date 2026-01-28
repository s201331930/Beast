"""
Comprehensive Backtesting Framework
====================================
Professional-grade backtesting system for validating anomaly detection signals:
- Event-based signal testing
- Walk-forward optimization
- Monte Carlo simulation
- Performance analytics
- Risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    signal_name: str
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_days: Optional[int] = None
    exit_reason: Optional[str] = None
    max_drawdown: Optional[float] = None
    max_profit: Optional[float] = None
    

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # Fraction of capital per trade
    max_positions: int = 5
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.20  # 20%
    trailing_stop: Optional[float] = None  # Optional trailing stop
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001  # 0.1%
    holding_period: int = 20  # Max days to hold
    min_holding_period: int = 1  # Min days before exit
    allow_short: bool = False
    

@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk Metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_period: float = 0.0
    
    # Advanced Metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    
    # Trade list
    trades: List[Trade] = field(default_factory=list)
    
    # Daily returns
    daily_returns: pd.Series = field(default_factory=pd.Series)


class SignalBacktester:
    """
    Event-driven backtester for anomaly detection signals.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results = None
        
    def backtest(self, data: pd.DataFrame, signals: pd.Series, 
                 signal_name: str = "signal") -> BacktestResults:
        """
        Run backtest on signals.
        
        Args:
            data: OHLCV data with 'close' column
            signals: Series of signals (1 for buy, -1 for sell, 0 for hold)
            signal_name: Name of the signal for reporting
        
        Returns:
            BacktestResults with comprehensive metrics
        """
        logger.info(f"Running backtest for '{signal_name}'...")
        
        # Initialize
        capital = self.config.initial_capital
        equity = [capital]
        trades = []
        open_positions = []
        
        # Get price data
        prices = data['close'].values
        dates = data.index
        
        for i in range(1, len(data)):
            current_price = prices[i]
            current_date = dates[i]
            prev_price = prices[i-1]
            
            # Update open positions
            positions_to_close = []
            for j, pos in enumerate(open_positions):
                # Calculate current P&L
                if pos.direction == 1:  # Long
                    current_pnl = (current_price - pos.entry_price) / pos.entry_price
                else:  # Short
                    current_pnl = (pos.entry_price - current_price) / pos.entry_price
                
                # Update max profit/drawdown
                if pos.max_profit is None or current_pnl > pos.max_profit:
                    pos.max_profit = current_pnl
                if pos.max_drawdown is None or current_pnl < pos.max_drawdown:
                    pos.max_drawdown = current_pnl
                
                holding_days = (current_date - pos.entry_date).days
                
                # Check exit conditions
                exit_reason = None
                
                # Stop loss
                if current_pnl <= -self.config.stop_loss:
                    exit_reason = "stop_loss"
                # Take profit
                elif current_pnl >= self.config.take_profit:
                    exit_reason = "take_profit"
                # Max holding period
                elif holding_days >= self.config.holding_period:
                    exit_reason = "max_holding"
                # Trailing stop
                elif self.config.trailing_stop and pos.max_profit:
                    if current_pnl < pos.max_profit - self.config.trailing_stop:
                        exit_reason = "trailing_stop"
                
                if exit_reason:
                    positions_to_close.append((j, exit_reason, current_pnl, holding_days))
            
            # Close positions
            for j, exit_reason, pnl, holding_days in sorted(positions_to_close, reverse=True):
                pos = open_positions.pop(j)
                
                # Apply slippage and commission
                exit_price = current_price * (1 - self.config.slippage * pos.direction)
                pnl_after_costs = pnl - 2 * self.config.commission
                
                pos.exit_date = current_date
                pos.exit_price = exit_price
                pos.pnl_percent = pnl_after_costs
                pos.pnl = pnl_after_costs * pos.size
                pos.holding_days = holding_days
                pos.exit_reason = exit_reason
                
                trades.append(pos)
                capital += pos.pnl
            
            # Check for new signals
            if i < len(signals) and signals.iloc[i] != 0:
                signal = signals.iloc[i]
                
                # Only take long if signal is positive (or short if allowed)
                if signal > 0 or (signal < 0 and self.config.allow_short):
                    # Check position limits
                    if len(open_positions) < self.config.max_positions:
                        # Calculate position size
                        position_value = capital * self.config.position_size
                        
                        # Apply slippage
                        entry_price = current_price * (1 + self.config.slippage * np.sign(signal))
                        
                        # Create new position
                        new_position = Trade(
                            entry_date=current_date,
                            entry_price=entry_price,
                            direction=int(np.sign(signal)),
                            size=position_value,
                            signal_name=signal_name
                        )
                        open_positions.append(new_position)
            
            equity.append(capital + sum(
                pos.size * ((current_price - pos.entry_price) / pos.entry_price * pos.direction)
                for pos in open_positions
            ))
        
        # Close any remaining positions at end
        final_price = prices[-1]
        final_date = dates[-1]
        for pos in open_positions:
            pnl = ((final_price - pos.entry_price) / pos.entry_price * pos.direction)
            pnl_after_costs = pnl - 2 * self.config.commission
            
            pos.exit_date = final_date
            pos.exit_price = final_price
            pos.pnl_percent = pnl_after_costs
            pos.pnl = pnl_after_costs * pos.size
            pos.holding_days = (final_date - pos.entry_date).days
            pos.exit_reason = "end_of_data"
            
            trades.append(pos)
        
        # Calculate results
        results = self._calculate_metrics(
            equity=pd.Series(equity, index=dates),
            trades=trades,
            benchmark=data['close']
        )
        
        self.results = results
        return results
    
    def _calculate_metrics(self, equity: pd.Series, trades: List[Trade],
                          benchmark: pd.Series) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        
        results = BacktestResults()
        results.trades = trades
        results.equity_curve = equity
        
        # Basic returns
        daily_returns = equity.pct_change().dropna()
        results.daily_returns = daily_returns
        
        total_days = len(equity)
        years = total_days / 252
        
        results.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        results.annualized_return = (1 + results.total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Benchmark
        results.benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        results.excess_return = results.total_return - results.benchmark_return
        
        # Volatility
        results.volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        results.sharpe_ratio = (results.annualized_return) / (results.volatility + 1e-10)
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        results.sortino_ratio = results.annualized_return / (downside_vol + 1e-10)
        
        # Drawdown Analysis
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        results.drawdown_curve = drawdown
        results.max_drawdown = drawdown.min()
        
        # Max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        for dd in in_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        results.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Calmar Ratio
        results.calmar_ratio = results.annualized_return / (abs(results.max_drawdown) + 1e-10)
        
        # Trade Statistics
        results.total_trades = len(trades)
        
        if trades:
            winning = [t for t in trades if t.pnl_percent and t.pnl_percent > 0]
            losing = [t for t in trades if t.pnl_percent and t.pnl_percent <= 0]
            
            results.winning_trades = len(winning)
            results.losing_trades = len(losing)
            results.win_rate = len(winning) / len(trades)
            
            results.avg_win = np.mean([t.pnl_percent for t in winning]) if winning else 0
            results.avg_loss = np.mean([t.pnl_percent for t in losing]) if losing else 0
            
            total_wins = sum(t.pnl_percent for t in winning) if winning else 0
            total_losses = abs(sum(t.pnl_percent for t in losing)) if losing else 0.001
            results.profit_factor = total_wins / total_losses
            
            results.avg_holding_period = np.mean([t.holding_days for t in trades if t.holding_days])
        
        # VaR and CVaR
        results.var_95 = np.percentile(daily_returns, 5)
        results.cvar_95 = daily_returns[daily_returns <= results.var_95].mean()
        
        # Tail Ratio
        right_tail = np.percentile(daily_returns, 95)
        left_tail = abs(np.percentile(daily_returns, 5))
        results.tail_ratio = right_tail / left_tail if left_tail > 0 else 1
        
        # Omega Ratio
        threshold = 0
        gains = daily_returns[daily_returns > threshold].sum()
        losses = abs(daily_returns[daily_returns < threshold].sum())
        results.omega_ratio = gains / losses if losses > 0 else float('inf')
        
        return results
    
    def generate_report(self, results: Optional[BacktestResults] = None) -> str:
        """Generate a detailed text report of backtest results."""
        
        results = results or self.results
        if results is None:
            return "No backtest results available."
        
        report = []
        report.append("=" * 60)
        report.append("           BACKTEST RESULTS REPORT")
        report.append("=" * 60)
        
        report.append("\n--- RETURN METRICS ---")
        report.append(f"Total Return:        {results.total_return * 100:.2f}%")
        report.append(f"Annualized Return:   {results.annualized_return * 100:.2f}%")
        report.append(f"Benchmark Return:    {results.benchmark_return * 100:.2f}%")
        report.append(f"Excess Return:       {results.excess_return * 100:.2f}%")
        
        report.append("\n--- RISK METRICS ---")
        report.append(f"Volatility (Ann.):   {results.volatility * 100:.2f}%")
        report.append(f"Sharpe Ratio:        {results.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio:       {results.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio:        {results.calmar_ratio:.3f}")
        report.append(f"Max Drawdown:        {results.max_drawdown * 100:.2f}%")
        report.append(f"Max DD Duration:     {results.max_drawdown_duration} days")
        
        report.append("\n--- TRADE STATISTICS ---")
        report.append(f"Total Trades:        {results.total_trades}")
        report.append(f"Winning Trades:      {results.winning_trades}")
        report.append(f"Losing Trades:       {results.losing_trades}")
        report.append(f"Win Rate:            {results.win_rate * 100:.1f}%")
        report.append(f"Avg Win:             {results.avg_win * 100:.2f}%")
        report.append(f"Avg Loss:            {results.avg_loss * 100:.2f}%")
        report.append(f"Profit Factor:       {results.profit_factor:.2f}")
        report.append(f"Avg Holding Period:  {results.avg_holding_period:.1f} days")
        
        report.append("\n--- ADVANCED METRICS ---")
        report.append(f"VaR (95%):           {results.var_95 * 100:.2f}%")
        report.append(f"CVaR (95%):          {results.cvar_95 * 100:.2f}%")
        report.append(f"Omega Ratio:         {results.omega_ratio:.2f}")
        report.append(f"Tail Ratio:          {results.tail_ratio:.2f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class WalkForwardOptimizer:
    """
    Walk-forward analysis for robust parameter optimization.
    """
    
    def __init__(self, train_period: int = 252, test_period: int = 63):
        self.train_period = train_period
        self.test_period = test_period
        
    def optimize(self, data: pd.DataFrame, signal_generator: Callable,
                param_grid: Dict[str, List], config: Optional[BacktestConfig] = None) -> Dict:
        """
        Perform walk-forward optimization.
        
        Args:
            data: Full dataset
            signal_generator: Function that takes (data, **params) and returns signals
            param_grid: Dictionary of parameter names to lists of values
            config: Backtesting configuration
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting walk-forward optimization...")
        
        config = config or BacktestConfig()
        backtester = SignalBacktester(config)
        
        results = []
        n_periods = (len(data) - self.train_period) // self.test_period
        
        for period in range(n_periods):
            train_start = period * self.test_period
            train_end = train_start + self.train_period
            test_end = train_end + self.test_period
            
            if test_end > len(data):
                break
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Find best parameters on training set
            best_params = None
            best_sharpe = -np.inf
            
            # Grid search (simplified)
            from itertools import product
            param_combinations = [dict(zip(param_grid.keys(), v)) 
                                 for v in product(*param_grid.values())]
            
            for params in param_combinations:
                try:
                    train_signals = signal_generator(train_data, **params)
                    train_results = backtester.backtest(train_data, train_signals)
                    
                    if train_results.sharpe_ratio > best_sharpe:
                        best_sharpe = train_results.sharpe_ratio
                        best_params = params
                except Exception as e:
                    logger.warning(f"Parameter combination failed: {e}")
                    continue
            
            # Test with best parameters
            if best_params:
                test_signals = signal_generator(test_data, **best_params)
                test_results = backtester.backtest(test_data, test_signals)
                
                results.append({
                    'period': period,
                    'train_start': data.index[train_start],
                    'test_start': data.index[train_end],
                    'test_end': data.index[min(test_end, len(data)-1)],
                    'best_params': best_params,
                    'train_sharpe': best_sharpe,
                    'test_sharpe': test_results.sharpe_ratio,
                    'test_return': test_results.total_return
                })
        
        # Aggregate results
        if results:
            avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
            avg_test_return = np.mean([r['test_return'] for r in results])
            
            return {
                'periods': results,
                'avg_test_sharpe': avg_test_sharpe,
                'avg_test_return': avg_test_return,
                'n_periods': len(results)
            }
        
        return {'periods': [], 'avg_test_sharpe': 0, 'avg_test_return': 0, 'n_periods': 0}


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        
    def simulate(self, trades: List[Trade]) -> Dict:
        """
        Run Monte Carlo simulation on trade results.
        
        Randomizes trade order to estimate distribution of possible outcomes.
        """
        if not trades:
            return {}
        
        # Extract trade returns
        trade_returns = [t.pnl_percent for t in trades if t.pnl_percent is not None]
        
        if len(trade_returns) < 5:
            return {}
        
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        simulated_returns = []
        simulated_drawdowns = []
        
        for _ in range(self.n_simulations):
            # Shuffle trade order
            shuffled = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + shuffled)
            total_return = cumulative[-1] - 1
            
            # Calculate max drawdown
            rolling_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            simulated_returns.append(total_return)
            simulated_drawdowns.append(max_dd)
        
        return {
            'mean_return': np.mean(simulated_returns),
            'median_return': np.median(simulated_returns),
            'std_return': np.std(simulated_returns),
            'return_5th_percentile': np.percentile(simulated_returns, 5),
            'return_95th_percentile': np.percentile(simulated_returns, 95),
            'mean_max_drawdown': np.mean(simulated_drawdowns),
            'worst_drawdown': min(simulated_drawdowns),
            'probability_positive': np.mean(np.array(simulated_returns) > 0)
        }


class EventStudyAnalyzer:
    """
    Event study analysis for signal quality assessment.
    Analyzes returns around signal events.
    """
    
    def __init__(self, pre_window: int = 10, post_window: int = 20):
        self.pre_window = pre_window
        self.post_window = post_window
        
    def analyze(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Analyze cumulative returns around signal events.
        """
        signal_dates = signals[signals != 0].index
        
        if len(signal_dates) == 0:
            return pd.DataFrame()
        
        returns = data['close'].pct_change()
        
        event_returns = []
        
        for date in signal_dates:
            try:
                date_idx = data.index.get_loc(date)
                
                start_idx = max(0, date_idx - self.pre_window)
                end_idx = min(len(data), date_idx + self.post_window + 1)
                
                window_returns = returns.iloc[start_idx:end_idx].values
                
                # Align to event date (event at index pre_window)
                aligned = np.full(self.pre_window + self.post_window + 1, np.nan)
                
                pre_available = date_idx - start_idx
                post_available = end_idx - date_idx - 1
                
                aligned_start = self.pre_window - pre_available
                aligned_end = self.pre_window + post_available + 1
                
                aligned[aligned_start:aligned_end] = window_returns
                
                event_returns.append(aligned)
                
            except Exception:
                continue
        
        if not event_returns:
            return pd.DataFrame()
        
        # Calculate average returns around events
        event_matrix = np.array(event_returns)
        
        avg_returns = np.nanmean(event_matrix, axis=0)
        std_returns = np.nanstd(event_matrix, axis=0)
        cumulative_returns = np.nancumprod(1 + avg_returns) - 1
        
        time_index = range(-self.pre_window, self.post_window + 1)
        
        result = pd.DataFrame({
            'day': time_index,
            'avg_return': avg_returns,
            'std_return': std_returns,
            'cumulative_return': cumulative_returns,
            'n_events': np.sum(~np.isnan(event_matrix), axis=0)
        })
        
        result.set_index('day', inplace=True)
        
        return result


if __name__ == "__main__":
    import yfinance as yf
    
    # Test backtester
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    
    # Create simple momentum signals
    returns = data['close'].pct_change()
    momentum = returns.rolling(20).mean()
    signals = pd.Series(0, index=data.index)
    signals[momentum > 0.01] = 1  # Buy signal
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.2,
        stop_loss=0.08,
        take_profit=0.25,
        holding_period=30
    )
    
    backtester = SignalBacktester(config)
    results = backtester.backtest(data, signals, "momentum_20d")
    
    print(backtester.generate_report())
    
    # Event study
    event_analyzer = EventStudyAnalyzer()
    event_results = event_analyzer.analyze(data, signals)
    print("\nEvent Study Results:")
    print(event_results)
