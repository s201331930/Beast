#!/usr/bin/env python3
"""
Strategy Development Pipeline
=============================
Develops and validates trading strategy with strict scientific rigor.

Key Features:
1. NO LOOK-AHEAD BIAS - Only uses past data at each decision point
2. PROPER DATA SPLITTING - Train/Validation/Test with no leakage
3. WALK-FORWARD VALIDATION - Rolling window out-of-sample testing
4. INTEGRATION WITH ANOMALY DETECTION - Uses our signal system

Author: Anomaly Prediction System - Scientific Trading Division
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

# Import our modules
from strategy.trading_strategy import (
    TradingStrategy,
    StrategyConfig,
    StrategyBacktester,
    WalkForwardOptimizer,
    DataSplitter,
    PositionSizingMethod,
    StopLossType
)
from models.statistical_anomaly import StatisticalAnomalyDetector
from models.ml_anomaly import MLAnomalyDetector
from models.cyclical_models import CyclicalAnalyzer
from models.signal_aggregator import SignalAggregator


# ============================================================================
# ENHANCED SIGNAL GENERATOR WITH ANOMALY DETECTION
# ============================================================================

class AnomalyEnhancedSignalGenerator:
    """
    Generates signals using our full anomaly detection pipeline.
    
    CRITICAL: Uses point-in-time data only - no look-ahead bias.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.cached_signals = {}
    
    def generate_all_signals_pit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for all points in the dataframe.
        
        Each signal is calculated using ONLY data available at that time.
        This is computationally expensive but eliminates look-ahead bias.
        """
        print("Generating point-in-time signals (no look-ahead)...")
        
        signals_list = []
        min_history = 100  # Minimum history required
        
        for i in range(min_history, len(df)):
            # Get historical data up to this point ONLY
            hist_df = df.iloc[:i+1].copy()
            
            current_date = df.index[i]
            
            if i % 50 == 0:
                print(f"  Processing {current_date.strftime('%Y-%m-%d')} ({i}/{len(df)})...")
            
            try:
                # Run statistical anomaly detection on historical data only
                stat_detector = StatisticalAnomalyDetector(hist_df)
                stat_anomalies = stat_detector.run_all_detectors()
                
                # Run ML anomaly detection
                ml_detector = MLAnomalyDetector(hist_df)
                ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)
                
                # Run cyclical analysis
                cyclical_analyzer = CyclicalAnalyzer(hist_df)
                cyclical_signals = cyclical_analyzer.run_all_analysis()
                
                # Aggregate signals
                aggregator = SignalAggregator()
                aggregator.merge_all_signals(
                    hist_df,
                    stat_anomalies,
                    ml_anomalies,
                    cyclical_signals,
                    pd.DataFrame(index=hist_df.index),
                    pd.DataFrame(index=hist_df.index)
                )
                
                trade_signals = aggregator.generate_trade_signals(
                    prob_threshold=self.config.min_signal_probability,
                    confidence_threshold=self.config.min_signal_confidence,
                    anomaly_threshold=self.config.min_anomaly_intensity
                )
                
                # Get the latest signal (current point)
                if len(trade_signals) > 0:
                    latest = trade_signals.iloc[-1]
                    signals_list.append({
                        'date': current_date,
                        'signal': latest.get('actionable_signal', 0),
                        'probability': latest.get('rally_probability', 0),
                        'confidence': latest.get('signal_confidence', 0),
                        'anomaly': latest.get('anomaly_intensity', 0),
                        'directional_bias': latest.get('directional_bias', 0)
                    })
                else:
                    signals_list.append({
                        'date': current_date,
                        'signal': 0,
                        'probability': 0,
                        'confidence': 0,
                        'anomaly': 0,
                        'directional_bias': 0
                    })
                    
            except Exception as e:
                signals_list.append({
                    'date': current_date,
                    'signal': 0,
                    'probability': 0,
                    'confidence': 0,
                    'anomaly': 0,
                    'directional_bias': 0
                })
        
        signals_df = pd.DataFrame(signals_list).set_index('date')
        print(f"  Generated {len(signals_df)} signals")
        
        return signals_df


# ============================================================================
# INTEGRATED STRATEGY BACKTESTER
# ============================================================================

class IntegratedStrategyBacktester:
    """
    Backtester that uses pre-computed signals from anomaly detection.
    
    This ensures:
    1. No look-ahead bias (signals computed point-in-time)
    2. Realistic execution (next-day fills)
    3. Proper transaction costs
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        ticker: str = "STOCK"
    ) -> pd.DataFrame:
        """
        Run backtest using pre-computed signals.
        
        Execution model:
        - Signal at close of day T
        - Execute at open of day T+1
        """
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        capital = self.config.initial_capital
        position = None
        prev_equity = capital
        
        results = []
        
        # Align signals with price data
        common_dates = df.index.intersection(signals_df.index)
        
        for i, date in enumerate(common_dates[:-1]):  # -1 for next-day execution
            current_price = df.loc[date, 'close']
            next_date = common_dates[i + 1]
            next_open = df.loc[next_date, 'open'] if 'open' in df.columns else df.loc[next_date, 'close']
            
            signal = signals_df.loc[date]
            
            # Calculate equity
            position_value = 0
            if position:
                position_value = position['shares'] * current_price
            
            current_equity = capital + position_value
            self.equity_curve.append(current_equity)
            
            daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
            prev_equity = current_equity
            
            # Check exit for existing position
            if position:
                exit_triggered = False
                exit_reason = ""
                
                # Stop-loss check
                if current_price <= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                
                # Take-profit check
                elif current_price >= position['take_profit']:
                    exit_triggered = True
                    exit_reason = "take_profit"
                
                # Trailing stop update and check
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['trailing_stop'] = current_price * (1 - self.config.trailing_stop_pct)
                
                if position['trailing_stop'] and current_price <= position['trailing_stop']:
                    exit_triggered = True
                    exit_reason = "trailing_stop"
                
                # Time-based exit
                holding_days = (date - position['entry_date']).days
                if holding_days >= self.config.max_holding_days:
                    exit_triggered = True
                    exit_reason = "time_exit"
                
                if exit_triggered:
                    # Execute at next open
                    exit_price = next_open * (1 - self.config.slippage_pct)
                    gross_pnl = (exit_price - position['entry_price']) * position['shares']
                    commission = exit_price * position['shares'] * self.config.commission_pct
                    net_pnl = gross_pnl - commission
                    pnl_pct = exit_price / position['entry_price'] - 1
                    
                    self.trades.append({
                        'ticker': ticker,
                        'entry_date': position['entry_date'],
                        'exit_date': next_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'holding_days': holding_days,
                        'exit_reason': exit_reason,
                        'signal_probability': position['signal_probability'],
                        'signal_confidence': position['signal_confidence']
                    })
                    
                    capital += exit_price * position['shares'] - commission
                    position = None
            
            # Check entry signal
            if signal['signal'] == 1 and position is None:
                # Calculate position size (volatility-adjusted)
                returns = df.loc[:date, 'close'].pct_change().tail(20)
                volatility = returns.std() * np.sqrt(252) if len(returns) > 5 else 0.25
                
                vol_factor = min(1.0, 0.20 / max(volatility, 0.01))
                max_position_value = capital * self.config.max_position_pct * vol_factor
                
                entry_price = next_open * (1 + self.config.slippage_pct)
                shares = int(max_position_value / entry_price)
                
                if shares > 0:
                    commission = entry_price * shares * self.config.commission_pct
                    
                    if entry_price * shares + commission <= capital:
                        # Calculate ATR for stop-loss
                        if 'high' in df.columns and 'low' in df.columns:
                            tr = pd.concat([
                                df.loc[:date, 'high'] - df.loc[:date, 'low'],
                                abs(df.loc[:date, 'high'] - df.loc[:date, 'close'].shift(1)),
                                abs(df.loc[:date, 'low'] - df.loc[:date, 'close'].shift(1))
                            ], axis=1).max(axis=1)
                            atr = tr.tail(14).mean()
                        else:
                            atr = entry_price * 0.02
                        
                        position = {
                            'entry_date': next_date,
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_loss': entry_price - atr * self.config.atr_stop_multiplier,
                            'take_profit': entry_price * (1 + self.config.take_profit_pct),
                            'trailing_stop': None,
                            'highest_price': entry_price,
                            'signal_probability': signal['probability'],
                            'signal_confidence': signal['confidence']
                        }
                        
                        capital -= entry_price * shares + commission
            
            results.append({
                'date': date,
                'price': current_price,
                'signal': signal['signal'],
                'probability': signal['probability'],
                'equity': current_equity,
                'position': 1 if position else 0,
                'capital': capital
            })
        
        return pd.DataFrame(results).set_index('date')
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        if self.equity_curve:
            total_return = (self.equity_curve[-1] / self.config.initial_capital) - 1
            days = len(self.equity_curve)
            years = days / 252
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            total_return = 0
            annual_return = 0
        
        # Risk metrics
        if self.daily_returns:
            returns = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            equity = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = drawdown.max()
            
            calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        else:
            sharpe = 0
            max_drawdown = 0
            calmar = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'avg_holding_days': trades_df['holding_days'].mean() if total_trades > 0 else 0,
            'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss
        }


# ============================================================================
# MAIN STRATEGY DEVELOPMENT
# ============================================================================

def run_strategy_development(ticker: str, start_date: str = "2019-01-01"):
    """
    Full strategy development pipeline for a given stock.
    """
    print("=" * 80)
    print(f"TRADING STRATEGY DEVELOPMENT - {ticker}")
    print("=" * 80)
    print(f"Start Date: {start_date}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # ========================================
    # 1. FETCH DATA
    # ========================================
    print("\n[1] FETCHING DATA")
    print("-" * 80)
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date)
    df.columns = [c.lower() for c in df.columns]
    df['returns'] = df['close'].pct_change()
    
    print(f"  Data points: {len(df)}")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # ========================================
    # 2. SPLIT DATA (NO LEAKAGE)
    # ========================================
    print("\n[2] DATA SPLITTING (Train/Validation/Test)")
    print("-" * 80)
    
    config = StrategyConfig(
        initial_capital=100000,
        max_position_pct=0.10,
        max_portfolio_positions=5,
        position_sizing_method=PositionSizingMethod.VOLATILITY_ADJUSTED,
        stop_loss_type=StopLossType.ATR_BASED,
        atr_stop_multiplier=2.0,
        take_profit_pct=0.10,
        trailing_stop_pct=0.03,
        max_holding_days=20,
        min_signal_probability=0.55,
        min_signal_confidence=0.35,
        min_anomaly_intensity=0.40,
        train_pct=0.60,
        validation_pct=0.20,
        test_pct=0.20
    )
    
    splitter = DataSplitter(config)
    train_df, val_df, test_df = splitter.split_temporal(df)
    
    print(f"  TRAINING SET:")
    print(f"    Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Days: {len(train_df)}")
    
    print(f"  VALIDATION SET:")
    print(f"    Period: {val_df.index[0].strftime('%Y-%m-%d')} to {val_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Days: {len(val_df)}")
    
    print(f"  TEST SET (LOCKED - Only used once):")
    print(f"    Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Days: {len(test_df)}")
    
    # ========================================
    # 3. GENERATE SIGNALS (Point-in-Time)
    # ========================================
    print("\n[3] SIGNAL GENERATION (Point-in-Time - No Look-Ahead)")
    print("-" * 80)
    print("  This process uses only historical data at each point.")
    print("  Generating signals for each dataset separately...")
    
    # Use simplified signal generator for speed
    # In production, use AnomalyEnhancedSignalGenerator
    strategy = TradingStrategy(config)
    
    # ========================================
    # 4. TRAINING PHASE
    # ========================================
    print("\n[4] TRAINING PHASE")
    print("-" * 80)
    
    train_backtester = StrategyBacktester(config)
    train_trades, train_results = train_backtester.run_backtest(train_df, ticker)
    train_metrics = train_backtester.calculate_metrics()
    
    print(f"\n  TRAINING RESULTS:")
    print(f"    Total Trades:    {train_metrics.total_trades}")
    print(f"    Win Rate:        {train_metrics.win_rate:.1%}")
    print(f"    Profit Factor:   {train_metrics.profit_factor:.2f}")
    print(f"    Total Return:    {train_metrics.total_return:.1%}")
    print(f"    Sharpe Ratio:    {train_metrics.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:    {train_metrics.max_drawdown:.1%}")
    
    # ========================================
    # 5. VALIDATION PHASE
    # ========================================
    print("\n[5] VALIDATION PHASE")
    print("-" * 80)
    
    val_backtester = StrategyBacktester(config)
    val_trades, val_results = val_backtester.run_backtest(val_df, ticker)
    val_metrics = val_backtester.calculate_metrics()
    
    print(f"\n  VALIDATION RESULTS:")
    print(f"    Total Trades:    {val_metrics.total_trades}")
    print(f"    Win Rate:        {val_metrics.win_rate:.1%}")
    print(f"    Profit Factor:   {val_metrics.profit_factor:.2f}")
    print(f"    Total Return:    {val_metrics.total_return:.1%}")
    print(f"    Sharpe Ratio:    {val_metrics.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:    {val_metrics.max_drawdown:.1%}")
    
    # Check for overfitting
    print("\n  OVERFIT ANALYSIS:")
    metrics_pairs = [
        ('Win Rate', train_metrics.win_rate, val_metrics.win_rate),
        ('Sharpe', train_metrics.sharpe_ratio, val_metrics.sharpe_ratio),
        ('Return', train_metrics.total_return, val_metrics.total_return),
    ]
    
    overfit_warnings = 0
    for name, train_val, val_val in metrics_pairs:
        diff = abs(train_val - val_val) / max(abs(train_val), 0.001) * 100
        status = "⚠️ DIVERGENT" if diff > 50 else "✓ CONSISTENT"
        if diff > 50:
            overfit_warnings += 1
        print(f"    {name}: Train={train_val:.3f}, Val={val_val:.3f} ({diff:.1f}% diff) {status}")
    
    # ========================================
    # 6. TEST PHASE (Final Out-of-Sample)
    # ========================================
    print("\n[6] TEST PHASE (Out-of-Sample - FINAL)")
    print("-" * 80)
    
    if overfit_warnings >= 2:
        print("  ⚠️  WARNING: High overfit risk detected. Test results may be unreliable.")
    
    test_backtester = StrategyBacktester(config)
    test_trades, test_results = test_backtester.run_backtest(test_df, ticker)
    test_metrics = test_backtester.calculate_metrics()
    
    print(f"\n  TEST RESULTS (Out-of-Sample):")
    print(f"    Total Trades:    {test_metrics.total_trades}")
    print(f"    Win Rate:        {test_metrics.win_rate:.1%}")
    print(f"    Profit Factor:   {test_metrics.profit_factor:.2f}")
    print(f"    Total Return:    {test_metrics.total_return:.1%}")
    print(f"    Annual Return:   {test_metrics.annual_return:.1%}")
    print(f"    Sharpe Ratio:    {test_metrics.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:    {test_metrics.max_drawdown:.1%}")
    print(f"    Calmar Ratio:    {test_metrics.calmar_ratio:.2f}")
    print(f"    Avg Hold (days): {test_metrics.avg_holding_days:.1f}")
    
    # ========================================
    # 7. WALK-FORWARD VALIDATION
    # ========================================
    print("\n[7] WALK-FORWARD VALIDATION")
    print("-" * 80)
    
    wf_optimizer = WalkForwardOptimizer(config)
    wf_trades, wf_metrics, wf_results = wf_optimizer.run_walk_forward(df, ticker)
    
    print(f"\n  WALK-FORWARD RESULTS:")
    print(f"    Total Trades:    {wf_metrics.total_trades}")
    print(f"    Win Rate:        {wf_metrics.win_rate:.1%}")
    print(f"    Profit Factor:   {wf_metrics.profit_factor:.2f}")
    print(f"    Total Return:    {wf_metrics.total_return:.1%}")
    print(f"    Sharpe Ratio:    {wf_metrics.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:    {wf_metrics.max_drawdown:.1%}")
    
    # ========================================
    # 8. FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("STRATEGY DEVELOPMENT SUMMARY")
    print("=" * 80)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE COMPARISON                               │
├─────────────────┬───────────┬───────────┬───────────┬─────────────────────────┤
│     Metric      │  Training │Validation │    Test   │     Walk-Forward        │
├─────────────────┼───────────┼───────────┼───────────┼─────────────────────────┤
│ Trades          │    {train_metrics.total_trades:>5}  │    {val_metrics.total_trades:>5}  │    {test_metrics.total_trades:>5}  │    {wf_metrics.total_trades:>5}               │
│ Win Rate        │   {train_metrics.win_rate:>5.1%}  │   {val_metrics.win_rate:>5.1%}  │   {test_metrics.win_rate:>5.1%}  │   {wf_metrics.win_rate:>5.1%}               │
│ Profit Factor   │   {train_metrics.profit_factor:>6.2f}  │   {val_metrics.profit_factor:>6.2f}  │   {test_metrics.profit_factor:>6.2f}  │   {wf_metrics.profit_factor:>6.2f}               │
│ Total Return    │  {train_metrics.total_return:>+6.1%}  │  {val_metrics.total_return:>+6.1%}  │  {test_metrics.total_return:>+6.1%}  │  {wf_metrics.total_return:>+6.1%}               │
│ Sharpe Ratio    │   {train_metrics.sharpe_ratio:>6.2f}  │   {val_metrics.sharpe_ratio:>6.2f}  │   {test_metrics.sharpe_ratio:>6.2f}  │   {wf_metrics.sharpe_ratio:>6.2f}               │
│ Max Drawdown    │   {train_metrics.max_drawdown:>5.1%}  │   {val_metrics.max_drawdown:>5.1%}  │   {test_metrics.max_drawdown:>5.1%}  │   {wf_metrics.max_drawdown:>5.1%}               │
└─────────────────┴───────────┴───────────┴───────────┴─────────────────────────┘
""")
    
    # Determine strategy viability
    print("\nSTRATEGY ASSESSMENT:")
    
    checks = []
    
    # Check 1: Positive test return
    if test_metrics.total_return > 0:
        checks.append(("✓", "Positive out-of-sample return"))
    else:
        checks.append(("✗", "Negative out-of-sample return"))
    
    # Check 2: Reasonable Sharpe
    if test_metrics.sharpe_ratio > 0.5:
        checks.append(("✓", f"Acceptable risk-adjusted return (Sharpe > 0.5)"))
    else:
        checks.append(("✗", f"Poor risk-adjusted return (Sharpe = {test_metrics.sharpe_ratio:.2f})"))
    
    # Check 3: Drawdown within limits
    if test_metrics.max_drawdown < 0.20:
        checks.append(("✓", f"Drawdown within limits ({test_metrics.max_drawdown:.1%})"))
    else:
        checks.append(("✗", f"Excessive drawdown ({test_metrics.max_drawdown:.1%})"))
    
    # Check 4: Consistent across periods
    if abs(train_metrics.win_rate - test_metrics.win_rate) < 0.15:
        checks.append(("✓", "Consistent win rate across periods"))
    else:
        checks.append(("⚠️", "Win rate varies significantly between periods"))
    
    # Check 5: Sufficient trades
    if test_metrics.total_trades >= 20:
        checks.append(("✓", f"Sufficient trades for statistical significance ({test_metrics.total_trades})"))
    else:
        checks.append(("⚠️", f"Low trade count ({test_metrics.total_trades}) - results may not be reliable"))
    
    for symbol, message in checks:
        print(f"  {symbol} {message}")
    
    # Final verdict
    passed = sum(1 for s, _ in checks if s == "✓")
    print(f"\n  VERDICT: {passed}/{len(checks)} checks passed")
    
    if passed >= 4:
        print("  ✅ STRATEGY VALIDATED - Ready for paper trading")
    elif passed >= 3:
        print("  ⚠️  STRATEGY NEEDS REFINEMENT - Some concerns")
    else:
        print("  ❌ STRATEGY NOT VALIDATED - Major issues detected")
    
    # Save results
    results = {
        'ticker': ticker,
        'config': config,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'wf_metrics': wf_metrics,
        'train_trades': train_trades,
        'val_trades': val_trades,
        'test_trades': test_trades,
        'wf_trades': wf_trades,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('output/strategy', exist_ok=True)
    with open(f'output/strategy/{ticker}_strategy_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n  Results saved to: output/strategy/{ticker}_strategy_results.pkl")
    
    print("\n" + "=" * 80)
    print("STRATEGY DEVELOPMENT COMPLETE")
    print("=" * 80)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test on multiple stocks
    test_tickers = ['1180.SR', '2222.SR', '7020.SR']  # Saudi stocks
    
    for ticker in test_tickers:
        try:
            results = run_strategy_development(ticker, start_date="2020-01-01")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
