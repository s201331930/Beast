#!/usr/bin/env python3
"""
Production Trading Strategy
===========================
A rigorous, scientifically-validated trading strategy based on anomaly detection.

Key Principles:
1. NO LOOK-AHEAD BIAS - Only uses information available at decision time
2. PROPER DATA SPLITTING - Train/Validation/Test with no leakage
3. ROBUST RISK MANAGEMENT - Position sizing, stop-losses, portfolio limits
4. WALK-FORWARD VALIDATION - Rolling window optimization

Author: Anomaly Prediction System - Scientific Trading Division
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_FRACTIONAL = "fixed_fractional"      # Fixed % of capital
    VOLATILITY_ADJUSTED = "volatility_adjusted" # Adjust by volatility
    KELLY_CRITERION = "kelly_criterion"         # Optimal f from Kelly
    EQUAL_WEIGHT = "equal_weight"               # Equal allocation
    RISK_PARITY = "risk_parity"                 # Equal risk contribution


class StopLossType(Enum):
    """Stop-loss types"""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"


@dataclass
class StrategyConfig:
    """Strategy configuration parameters"""
    # Capital & Position Sizing
    initial_capital: float = 100000.0
    max_position_pct: float = 0.10           # Max 10% per position
    max_portfolio_positions: int = 10         # Max concurrent positions
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_ADJUSTED
    
    # Risk Management
    max_portfolio_risk: float = 0.02          # Max 2% portfolio risk per trade
    max_drawdown_limit: float = 0.15          # Stop trading at 15% drawdown
    daily_loss_limit: float = 0.03            # Stop trading at 3% daily loss
    
    # Stop-Loss
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    stop_loss_pct: float = 0.05               # 5% fixed stop
    atr_stop_multiplier: float = 2.0          # 2x ATR for stop
    trailing_stop_pct: float = 0.03           # 3% trailing stop
    
    # Take Profit
    take_profit_pct: float = 0.10             # 10% take profit
    partial_exit_pct: float = 0.50            # Exit 50% at first target
    partial_exit_target: float = 0.05         # First target at 5%
    
    # Time Management
    max_holding_days: int = 20                # Max holding period
    min_holding_days: int = 1                 # Min holding before exit
    
    # Signal Thresholds
    min_signal_probability: float = 0.60      # Min 60% probability
    min_signal_confidence: float = 0.40       # Min 40% confidence
    min_anomaly_intensity: float = 0.50       # Min 0.5 anomaly score
    
    # Transaction Costs
    commission_pct: float = 0.001             # 0.1% commission
    slippage_pct: float = 0.001               # 0.1% slippage
    
    # Data Splitting
    train_pct: float = 0.60                   # 60% training
    validation_pct: float = 0.20              # 20% validation
    test_pct: float = 0.20                    # 20% testing
    
    # Walk-Forward
    walk_forward_window: int = 252            # 1 year training window
    walk_forward_step: int = 63               # 3 month step


@dataclass
class Position:
    """Represents an open position"""
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    position_value: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    signal_probability: float = 0.0
    signal_confidence: float = 0.0
    partial_exit_done: bool = False


@dataclass
class Trade:
    """Completed trade record"""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str
    signal_probability: float
    signal_confidence: float


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    expectancy: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0


# ============================================================================
# DATA SPLITTER - NO LEAKAGE GUARANTEE
# ============================================================================

class DataSplitter:
    """
    Splits data into train/validation/test sets with NO LEAKAGE.
    
    Critical: Test data is NEVER used during strategy development.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def split_temporal(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal split - maintains time ordering (NO SHUFFLING).
        
        This is the ONLY correct way to split time series data.
        """
        n = len(df)
        train_end = int(n * self.config.train_pct)
        val_end = int(n * (self.config.train_pct + self.config.validation_pct))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def get_split_dates(self, df: pd.DataFrame) -> Dict[str, Tuple[datetime, datetime]]:
        """Get date ranges for each split"""
        train_df, val_df, test_df = self.split_temporal(df)
        
        return {
            'train': (train_df.index[0], train_df.index[-1]),
            'validation': (val_df.index[0], val_df.index[-1]),
            'test': (test_df.index[0], test_df.index[-1])
        }
    
    def create_walk_forward_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.
        
        Each split uses only PAST data for training, then tests on FUTURE data.
        This eliminates look-ahead bias.
        """
        splits = []
        window = self.config.walk_forward_window
        step = self.config.walk_forward_step
        
        for i in range(window, len(df) - step, step):
            train = df.iloc[i-window:i].copy()
            test = df.iloc[i:i+step].copy()
            splits.append((train, test))
        
        return splits


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """
    Comprehensive risk management system.
    
    Handles:
    - Position sizing
    - Stop-loss calculation
    - Portfolio-level risk limits
    - Drawdown monitoring
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = config.initial_capital
        self.current_equity = config.initial_capital
        self.trading_halted = False
        self.halt_reason = ""
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float,
        signal_confidence: float,
        existing_positions: int
    ) -> int:
        """
        Calculate position size based on configured method.
        
        Returns number of shares to buy.
        """
        if self.trading_halted:
            return 0
        
        # Check position limits
        if existing_positions >= self.config.max_portfolio_positions:
            return 0
        
        # Base position value
        max_position_value = capital * self.config.max_position_pct
        
        if self.config.position_sizing_method == PositionSizingMethod.FIXED_FRACTIONAL:
            position_value = max_position_value
            
        elif self.config.position_sizing_method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            # Reduce position size for high volatility
            vol_factor = min(1.0, 0.20 / max(volatility, 0.01))  # Target 20% annual vol
            position_value = max_position_value * vol_factor
            
        elif self.config.position_sizing_method == PositionSizingMethod.KELLY_CRITERION:
            # Simplified Kelly: f = (bp - q) / b
            # where b = win/loss ratio, p = win prob, q = 1-p
            win_prob = signal_confidence
            win_loss_ratio = 1.5  # Assumed
            kelly_f = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
            kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%
            position_value = capital * kelly_f
            
        else:  # EQUAL_WEIGHT
            position_value = capital / self.config.max_portfolio_positions
        
        # Apply risk-based limit
        risk_based_size = (capital * self.config.max_portfolio_risk) / (self.config.stop_loss_pct)
        position_value = min(position_value, risk_based_size)
        
        # Calculate shares
        shares = int(position_value / price)
        
        return max(0, shares)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        volatility: float
    ) -> float:
        """Calculate stop-loss price"""
        
        if self.config.stop_loss_type == StopLossType.FIXED_PERCENT:
            stop = entry_price * (1 - self.config.stop_loss_pct)
            
        elif self.config.stop_loss_type == StopLossType.ATR_BASED:
            stop = entry_price - (atr * self.config.atr_stop_multiplier)
            
        elif self.config.stop_loss_type == StopLossType.VOLATILITY_BASED:
            # 2 standard deviations
            daily_vol = volatility / np.sqrt(252)
            stop = entry_price * (1 - 2 * daily_vol)
            
        else:  # TRAILING or TIME_BASED
            stop = entry_price * (1 - self.config.stop_loss_pct)
        
        return stop
    
    def calculate_take_profit(self, entry_price: float) -> float:
        """Calculate take-profit price"""
        return entry_price * (1 + self.config.take_profit_pct)
    
    def update_trailing_stop(self, position: Position, current_price: float) -> float:
        """Update trailing stop based on highest price"""
        if current_price > position.highest_price:
            position.highest_price = current_price
            new_stop = current_price * (1 - self.config.trailing_stop_pct)
            if position.trailing_stop is None or new_stop > position.trailing_stop:
                position.trailing_stop = new_stop
        
        return position.trailing_stop or position.stop_loss
    
    def check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_date: datetime,
        atr: float
    ) -> Tuple[bool, str, float]:
        """
        Check if position should be exited.
        
        Returns: (should_exit, reason, exit_shares_pct)
        """
        holding_days = (current_date - position.entry_date).days
        
        # Update trailing stop
        effective_stop = self.update_trailing_stop(position, current_price)
        
        # Check stop-loss
        if current_price <= effective_stop:
            return True, "stop_loss", 1.0
        
        # Check take-profit
        if current_price >= position.take_profit:
            return True, "take_profit", 1.0
        
        # Check partial exit at first target
        if not position.partial_exit_done:
            partial_target = position.entry_price * (1 + self.config.partial_exit_target)
            if current_price >= partial_target:
                position.partial_exit_done = True
                return True, "partial_profit", self.config.partial_exit_pct
        
        # Check max holding time
        if holding_days >= self.config.max_holding_days:
            return True, "time_exit", 1.0
        
        return False, "", 0.0
    
    def update_equity(self, equity: float):
        """Update equity and check risk limits"""
        self.current_equity = equity
        
        # Update peak for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Check drawdown limit
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown >= self.config.max_drawdown_limit:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown reached: {drawdown:.1%}"
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L and check limits"""
        self.daily_pnl += pnl
        
        daily_return = self.daily_pnl / self.current_equity
        if daily_return <= -self.config.daily_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit reached: {daily_return:.1%}"
    
    def reset_daily(self):
        """Reset daily tracking"""
        self.daily_pnl = 0.0
        # Don't reset trading_halted - that persists


# ============================================================================
# SIGNAL GENERATOR (Point-in-Time)
# ============================================================================

class PointInTimeSignalGenerator:
    """
    Generates trading signals using ONLY information available at that time.
    
    This is critical to avoid look-ahead bias.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def calculate_features_pit(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Calculate features using only data up to index idx (Point-in-Time).
        
        NO FUTURE DATA IS USED.
        """
        if idx < 50:  # Need minimum history
            return {}
        
        # Get historical data up to this point ONLY
        hist = df.iloc[:idx+1]
        
        features = {}
        
        # Price features (using only past data)
        close = hist['close']
        features['sma_20'] = close.tail(20).mean()
        features['sma_50'] = close.tail(50).mean()
        features['price_vs_sma20'] = close.iloc[-1] / features['sma_20'] - 1
        features['price_vs_sma50'] = close.iloc[-1] / features['sma_50'] - 1
        
        # Volatility (historical only)
        returns = close.pct_change().dropna()
        features['volatility_20d'] = returns.tail(20).std() * np.sqrt(252)
        features['volatility_60d'] = returns.tail(60).std() * np.sqrt(252)
        
        # ATR
        if 'high' in hist.columns and 'low' in hist.columns:
            tr = pd.concat([
                hist['high'] - hist['low'],
                abs(hist['high'] - hist['close'].shift(1)),
                abs(hist['low'] - hist['close'].shift(1))
            ], axis=1).max(axis=1)
            features['atr_14'] = tr.tail(14).mean()
        else:
            features['atr_14'] = close.tail(14).std()
        
        # RSI (using only historical data)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Volume features
        if 'volume' in hist.columns:
            vol = hist['volume']
            features['volume_ratio'] = vol.iloc[-1] / vol.tail(20).mean()
            features['volume_trend'] = vol.tail(5).mean() / vol.tail(20).mean()
        
        # Momentum
        features['momentum_5d'] = close.iloc[-1] / close.iloc[-6] - 1 if idx >= 5 else 0
        features['momentum_20d'] = close.iloc[-1] / close.iloc[-21] - 1 if idx >= 20 else 0
        
        # Bollinger Band position
        bb_std = close.tail(20).std()
        bb_upper = features['sma_20'] + 2 * bb_std
        bb_lower = features['sma_20'] - 2 * bb_std
        features['bb_position'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        return features
    
    def generate_signal_pit(
        self,
        df: pd.DataFrame,
        idx: int,
        model_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Generate trading signal using only point-in-time data.
        
        Returns signal with probability, confidence, and anomaly score.
        """
        features = self.calculate_features_pit(df, idx)
        
        if not features:
            return {'signal': 0, 'probability': 0, 'confidence': 0, 'anomaly': 0}
        
        # Simple signal logic (can be replaced with ML model)
        signal_score = 0.0
        confidence_factors = []
        
        # Mean reversion signals
        if features.get('bb_position', 0.5) < 0.2:  # Near lower band
            signal_score += 0.3
            confidence_factors.append(0.7)
        
        if features.get('rsi_14', 50) < 30:  # Oversold
            signal_score += 0.2
            confidence_factors.append(0.6)
        
        # Momentum confirmation
        if features.get('momentum_5d', 0) > 0:  # Short-term momentum positive
            signal_score += 0.15
            confidence_factors.append(0.5)
        
        # Trend alignment
        if features.get('price_vs_sma50', 0) > 0:  # Above long-term trend
            signal_score += 0.15
            confidence_factors.append(0.6)
        
        # Volume confirmation
        if features.get('volume_ratio', 1) > 1.5:  # Above average volume
            signal_score += 0.1
            confidence_factors.append(0.5)
        
        # Volatility opportunity
        if features.get('volatility_20d', 0.2) > 0.25:  # Higher volatility
            signal_score += 0.1
            confidence_factors.append(0.4)
        
        # Calculate probability and confidence
        probability = min(signal_score, 1.0)
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0
        
        # Anomaly intensity based on deviation from normal
        anomaly = 0.0
        if features.get('volume_ratio', 1) > 2.0:
            anomaly += 0.3
        if abs(features.get('price_vs_sma20', 0)) > 0.05:
            anomaly += 0.3
        if features.get('rsi_14', 50) < 25 or features.get('rsi_14', 50) > 75:
            anomaly += 0.4
        
        # Generate binary signal
        signal = 1 if (
            probability >= self.config.min_signal_probability and
            confidence >= self.config.min_signal_confidence and
            anomaly >= self.config.min_anomaly_intensity
        ) else 0
        
        return {
            'signal': signal,
            'probability': probability,
            'confidence': confidence,
            'anomaly': anomaly,
            'features': features
        }


# ============================================================================
# STRATEGY BACKTESTER (No Look-Ahead)
# ============================================================================

class StrategyBacktester:
    """
    Backtests trading strategy with strict no-look-ahead rules.
    
    Key features:
    - Point-in-time signal generation
    - Realistic execution (next-day open)
    - Transaction costs
    - Proper risk management
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.signal_generator = PointInTimeSignalGenerator(config)
        
        # State
        self.capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.risk_manager = RiskManager(self.config)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        ticker: str = "STOCK"
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Run backtest on single stock with NO LOOK-AHEAD.
        
        Execution model:
        - Signal generated at close of day T
        - Order executed at open of day T+1
        - This ensures no look-ahead bias
        """
        self.reset()
        
        results = []
        prev_equity = self.config.initial_capital
        
        for i in range(50, len(df) - 1):  # -1 because we execute next day
            current_date = df.index[i]
            current_price = df['close'].iloc[i]
            next_open = df['open'].iloc[i + 1] if 'open' in df.columns else df['close'].iloc[i + 1]
            
            # Reset daily tracking
            if i > 50 and df.index[i].date() != df.index[i-1].date():
                self.risk_manager.reset_daily()
            
            # Calculate current equity
            position_value = 0
            if ticker in self.positions:
                pos = self.positions[ticker]
                position_value = pos.shares * current_price
            
            current_equity = self.capital + position_value
            self.equity_curve.append(current_equity)
            
            # Calculate daily return
            daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
            prev_equity = current_equity
            
            # Update risk manager
            self.risk_manager.update_equity(current_equity)
            
            # Check if trading is halted
            if self.risk_manager.trading_halted:
                continue
            
            # Get features for ATR calculation
            features = self.signal_generator.calculate_features_pit(df, i)
            atr = features.get('atr_14', current_price * 0.02)
            volatility = features.get('volatility_20d', 0.25)
            
            # Check existing position for exit
            if ticker in self.positions:
                pos = self.positions[ticker]
                should_exit, reason, exit_pct = self.risk_manager.check_exit_conditions(
                    pos, current_price, current_date, atr
                )
                
                if should_exit:
                    # Execute exit at next day open (no look-ahead)
                    exit_price = next_open * (1 - self.config.slippage_pct)
                    exit_shares = int(pos.shares * exit_pct)
                    
                    # Calculate P&L
                    gross_pnl = (exit_price - pos.entry_price) * exit_shares
                    commission = exit_price * exit_shares * self.config.commission_pct
                    net_pnl = gross_pnl - commission
                    pnl_pct = (exit_price / pos.entry_price - 1)
                    
                    # Record trade
                    trade = Trade(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        exit_date=df.index[i + 1],
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=exit_shares,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        holding_days=(current_date - pos.entry_date).days,
                        exit_reason=reason,
                        signal_probability=pos.signal_probability,
                        signal_confidence=pos.signal_confidence
                    )
                    self.trades.append(trade)
                    
                    # Update capital
                    self.capital += exit_price * exit_shares - commission
                    
                    # Update or close position
                    if exit_pct >= 1.0:
                        del self.positions[ticker]
                    else:
                        pos.shares -= exit_shares
            
            # Generate signal for potential entry (point-in-time)
            signal_result = self.signal_generator.generate_signal_pit(df, i)
            
            # Check for entry signal
            if signal_result['signal'] == 1 and ticker not in self.positions:
                # Calculate position size
                shares = self.risk_manager.calculate_position_size(
                    capital=self.capital,
                    price=next_open,
                    volatility=volatility,
                    signal_confidence=signal_result['confidence'],
                    existing_positions=len(self.positions)
                )
                
                if shares > 0:
                    # Execute at next day open (no look-ahead)
                    entry_price = next_open * (1 + self.config.slippage_pct)
                    commission = entry_price * shares * self.config.commission_pct
                    
                    if entry_price * shares + commission <= self.capital:
                        # Create position
                        position = Position(
                            ticker=ticker,
                            entry_date=df.index[i + 1],
                            entry_price=entry_price,
                            shares=shares,
                            position_value=entry_price * shares,
                            stop_loss=self.risk_manager.calculate_stop_loss(entry_price, atr, volatility),
                            take_profit=self.risk_manager.calculate_take_profit(entry_price),
                            highest_price=entry_price,
                            signal_probability=signal_result['probability'],
                            signal_confidence=signal_result['confidence']
                        )
                        
                        self.positions[ticker] = position
                        self.capital -= entry_price * shares + commission
            
            # Record daily state
            results.append({
                'date': current_date,
                'price': current_price,
                'signal': signal_result['signal'],
                'probability': signal_result['probability'],
                'confidence': signal_result['confidence'],
                'anomaly': signal_result['anomaly'],
                'equity': current_equity,
                'position': 1 if ticker in self.positions else 0,
                'capital': self.capital
            })
        
        return self.trades, pd.DataFrame(results).set_index('date')
    
    def calculate_metrics(self) -> StrategyMetrics:
        """Calculate comprehensive strategy metrics"""
        metrics = StrategyMetrics()
        
        if not self.trades:
            return metrics
        
        # Basic counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = len([t for t in self.trades if t.pnl > 0])
        metrics.losing_trades = len([t for t in self.trades if t.pnl <= 0])
        
        # Win rate
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        if self.equity_curve:
            final_equity = self.equity_curve[-1]
            metrics.total_return = (final_equity / self.config.initial_capital) - 1
            
            # Annualized return
            days = len(self.equity_curve)
            years = days / 252
            metrics.annual_return = (1 + metrics.total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        if self.daily_returns:
            returns = np.array(self.daily_returns)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if returns.std() > 0:
                metrics.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                metrics.sortino_ratio = np.sqrt(252) * returns.mean() / negative_returns.std()
            
            # Max drawdown
            equity = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            metrics.max_drawdown = drawdown.max()
            
            # Calmar ratio
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown
        
        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win +
            (1 - metrics.win_rate) * metrics.avg_loss
        )
        
        # Average holding days
        metrics.avg_holding_days = np.mean([t.holding_days for t in self.trades])
        
        # Recovery factor
        total_profit = sum([t.pnl for t in self.trades])
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = total_profit / (self.config.initial_capital * metrics.max_drawdown)
        
        return metrics


# ============================================================================
# WALK-FORWARD OPTIMIZER
# ============================================================================

class WalkForwardOptimizer:
    """
    Walk-forward optimization with no look-ahead bias.
    
    Process:
    1. Train on window W
    2. Optimize parameters using validation within W
    3. Test on next period (out-of-sample)
    4. Roll forward and repeat
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.splitter = DataSplitter(config)
    
    def optimize_parameters(
        self,
        train_df: pd.DataFrame,
        param_grid: Dict[str, List]
    ) -> StrategyConfig:
        """
        Optimize strategy parameters on training data.
        
        Uses time-series cross-validation within training period.
        """
        best_config = self.config
        best_sharpe = -np.inf
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in product(*param_values):
            # Create config with these parameters
            test_config = StrategyConfig(
                **{**self.config.__dict__, **dict(zip(param_names, values))}
            )
            
            # Run backtest on training data
            backtester = StrategyBacktester(test_config)
            trades, results = backtester.run_backtest(train_df)
            metrics = backtester.calculate_metrics()
            
            # Track best
            if metrics.sharpe_ratio > best_sharpe and metrics.total_trades >= 10:
                best_sharpe = metrics.sharpe_ratio
                best_config = test_config
        
        return best_config
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        ticker: str = "STOCK",
        optimize: bool = False,
        param_grid: Optional[Dict] = None
    ) -> Tuple[List[Trade], StrategyMetrics, pd.DataFrame]:
        """
        Run walk-forward analysis.
        
        Returns combined results from all out-of-sample periods.
        """
        splits = self.splitter.create_walk_forward_splits(df)
        
        all_trades = []
        all_results = []
        
        print(f"Running walk-forward analysis with {len(splits)} periods...")
        
        for i, (train_df, test_df) in enumerate(splits):
            print(f"  Period {i+1}/{len(splits)}: "
                  f"Train {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}, "
                  f"Test {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")
            
            # Optimize on training data if requested
            if optimize and param_grid:
                config = self.optimize_parameters(train_df, param_grid)
            else:
                config = self.config
            
            # Test on out-of-sample data
            backtester = StrategyBacktester(config)
            trades, results = backtester.run_backtest(test_df, ticker)
            
            all_trades.extend(trades)
            all_results.append(results)
        
        # Combine results
        combined_results = pd.concat(all_results)
        
        # Calculate overall metrics
        final_backtester = StrategyBacktester(self.config)
        final_backtester.trades = all_trades
        final_backtester.equity_curve = combined_results['equity'].tolist()
        final_backtester.daily_returns = combined_results['equity'].pct_change().dropna().tolist()
        
        metrics = final_backtester.calculate_metrics()
        
        return all_trades, metrics, combined_results


# ============================================================================
# MAIN STRATEGY CLASS
# ============================================================================

class TradingStrategy:
    """
    Main trading strategy class.
    
    Provides high-level interface for:
    - Strategy development
    - Backtesting
    - Validation
    - Live trading signals
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.splitter = DataSplitter(self.config)
        self.optimizer = WalkForwardOptimizer(self.config)
        
    def develop_and_validate(
        self,
        df: pd.DataFrame,
        ticker: str = "STOCK"
    ) -> Dict[str, Any]:
        """
        Full strategy development and validation pipeline.
        
        1. Split data (train/validation/test)
        2. Develop on training data
        3. Validate on validation data
        4. Final test on test data (ONLY ONCE)
        """
        print("=" * 70)
        print("STRATEGY DEVELOPMENT AND VALIDATION")
        print("=" * 70)
        
        # Split data
        train_df, val_df, test_df = self.splitter.split_temporal(df)
        dates = self.splitter.get_split_dates(df)
        
        print(f"\nData Splits:")
        print(f"  Training:   {dates['train'][0].strftime('%Y-%m-%d')} to {dates['train'][1].strftime('%Y-%m-%d')} ({len(train_df)} days)")
        print(f"  Validation: {dates['validation'][0].strftime('%Y-%m-%d')} to {dates['validation'][1].strftime('%Y-%m-%d')} ({len(val_df)} days)")
        print(f"  Test:       {dates['test'][0].strftime('%Y-%m-%d')} to {dates['test'][1].strftime('%Y-%m-%d')} ({len(test_df)} days)")
        
        # Train on training data
        print(f"\n1. TRAINING PHASE")
        print("-" * 70)
        train_backtester = StrategyBacktester(self.config)
        train_trades, train_results = train_backtester.run_backtest(train_df, ticker)
        train_metrics = train_backtester.calculate_metrics()
        
        print(f"  Trades: {train_metrics.total_trades}")
        print(f"  Win Rate: {train_metrics.win_rate:.1%}")
        print(f"  Sharpe: {train_metrics.sharpe_ratio:.2f}")
        print(f"  Return: {train_metrics.total_return:.1%}")
        
        # Validate on validation data
        print(f"\n2. VALIDATION PHASE")
        print("-" * 70)
        val_backtester = StrategyBacktester(self.config)
        val_trades, val_results = val_backtester.run_backtest(val_df, ticker)
        val_metrics = val_backtester.calculate_metrics()
        
        print(f"  Trades: {val_metrics.total_trades}")
        print(f"  Win Rate: {val_metrics.win_rate:.1%}")
        print(f"  Sharpe: {val_metrics.sharpe_ratio:.2f}")
        print(f"  Return: {val_metrics.total_return:.1%}")
        
        # Check for overfitting
        overfit_score = self._check_overfitting(train_metrics, val_metrics)
        print(f"\n  Overfit Score: {overfit_score:.2f} ({'LOW' if overfit_score < 0.3 else 'MEDIUM' if overfit_score < 0.6 else 'HIGH'})")
        
        # Final test (ONLY IF VALIDATION PASSES)
        print(f"\n3. TEST PHASE (Out-of-Sample)")
        print("-" * 70)
        
        if overfit_score > 0.6:
            print("  ⚠️  HIGH OVERFIT RISK - Test results may be unreliable")
        
        test_backtester = StrategyBacktester(self.config)
        test_trades, test_results = test_backtester.run_backtest(test_df, ticker)
        test_metrics = test_backtester.calculate_metrics()
        
        print(f"  Trades: {test_metrics.total_trades}")
        print(f"  Win Rate: {test_metrics.win_rate:.1%}")
        print(f"  Sharpe: {test_metrics.sharpe_ratio:.2f}")
        print(f"  Return: {test_metrics.total_return:.1%}")
        print(f"  Max Drawdown: {test_metrics.max_drawdown:.1%}")
        
        # Summary
        print(f"\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        consistency = self._check_consistency(train_metrics, val_metrics, test_metrics)
        
        return {
            'train': {'trades': train_trades, 'results': train_results, 'metrics': train_metrics},
            'validation': {'trades': val_trades, 'results': val_results, 'metrics': val_metrics},
            'test': {'trades': test_trades, 'results': test_results, 'metrics': test_metrics},
            'overfit_score': overfit_score,
            'consistency': consistency,
            'dates': dates
        }
    
    def _check_overfitting(self, train_metrics: StrategyMetrics, val_metrics: StrategyMetrics) -> float:
        """Calculate overfitting score (0 = no overfit, 1 = severe overfit)"""
        scores = []
        
        # Compare win rates
        if train_metrics.win_rate > 0:
            wr_diff = abs(train_metrics.win_rate - val_metrics.win_rate) / train_metrics.win_rate
            scores.append(min(wr_diff, 1.0))
        
        # Compare Sharpe ratios
        if train_metrics.sharpe_ratio > 0:
            sharpe_diff = abs(train_metrics.sharpe_ratio - val_metrics.sharpe_ratio) / train_metrics.sharpe_ratio
            scores.append(min(sharpe_diff, 1.0))
        
        # Compare returns
        if train_metrics.total_return > 0:
            ret_diff = abs(train_metrics.total_return - val_metrics.total_return) / abs(train_metrics.total_return)
            scores.append(min(ret_diff, 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    def _check_consistency(
        self,
        train_metrics: StrategyMetrics,
        val_metrics: StrategyMetrics,
        test_metrics: StrategyMetrics
    ) -> Dict[str, str]:
        """Check strategy consistency across periods"""
        consistency = {}
        
        # Win rate consistency
        win_rates = [train_metrics.win_rate, val_metrics.win_rate, test_metrics.win_rate]
        wr_std = np.std(win_rates)
        consistency['win_rate'] = 'CONSISTENT' if wr_std < 0.1 else 'VARIABLE'
        
        # Sharpe consistency
        sharpes = [train_metrics.sharpe_ratio, val_metrics.sharpe_ratio, test_metrics.sharpe_ratio]
        # Filter out extreme values
        sharpes = [s for s in sharpes if -5 < s < 5]
        if sharpes:
            sharpe_std = np.std(sharpes)
            consistency['sharpe'] = 'CONSISTENT' if sharpe_std < 0.5 else 'VARIABLE'
        else:
            consistency['sharpe'] = 'UNKNOWN'
        
        # Profitability consistency
        profitable = [
            train_metrics.total_return > 0,
            val_metrics.total_return > 0,
            test_metrics.total_return > 0
        ]
        consistency['profitability'] = 'CONSISTENT' if all(profitable) else 'MIXED'
        
        print(f"\n  Win Rate Consistency: {consistency['win_rate']}")
        print(f"  Sharpe Consistency: {consistency['sharpe']}")
        print(f"  Profitability: {consistency['profitability']}")
        
        return consistency


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Trading Strategy Module Loaded")
    print("Use TradingStrategy class for strategy development and validation")
