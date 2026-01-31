#!/usr/bin/env python3
"""
FULL PORTFOLIO SIMULATION - TASI Market
========================================
Comprehensive validation of the trading system across ALL Saudi stocks.

This simulation:
1. Screens all TASI stocks using our scoring system
2. Applies the trading strategy with full risk management
3. Simulates a real portfolio with position limits
4. Tracks P&L across different time periods
5. Provides comprehensive statistics

NO LOOK-AHEAD BIAS - All decisions made with point-in-time data only.

Author: Anomaly Prediction System - Scientific Trading Division
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pickle
import json

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PortfolioConfig:
    """Portfolio simulation configuration"""
    # Capital
    initial_capital: float = 1_000_000  # 1 Million SAR
    
    # Position Limits
    max_position_pct: float = 0.05       # Max 5% per position
    max_positions: int = 20               # Max 20 positions
    min_position_value: float = 10000     # Min 10K SAR per position
    
    # Risk Management
    max_portfolio_drawdown: float = 0.15  # 15% max drawdown
    max_sector_exposure: float = 0.30     # 30% max per sector
    max_single_stock_risk: float = 0.02   # 2% risk per trade
    
    # Entry Rules
    min_screening_score: float = 55       # Minimum screening score
    min_signal_probability: float = 0.55  # Minimum signal probability
    min_signal_confidence: float = 0.35   # Minimum confidence
    
    # Exit Rules
    stop_loss_atr_mult: float = 2.0       # 2x ATR stop-loss
    take_profit_pct: float = 0.10         # 10% take profit
    trailing_stop_pct: float = 0.03       # 3% trailing stop
    max_holding_days: int = 20            # Max 20 days hold
    
    # Transaction Costs
    commission_pct: float = 0.001         # 0.1% commission
    slippage_pct: float = 0.001           # 0.1% slippage
    
    # Rebalancing
    rebalance_frequency: str = 'daily'    # daily signal check


# Complete TASI Stock Universe with Sectors
TASI_UNIVERSE = {
    # Banking
    '1180.SR': {'name': 'Al Rajhi Bank', 'sector': 'Banking'},
    '1010.SR': {'name': 'Riyad Bank', 'sector': 'Banking'},
    '1050.SR': {'name': 'Banque Saudi Fransi', 'sector': 'Banking'},
    '1060.SR': {'name': 'Saudi Awwal Bank', 'sector': 'Banking'},
    '1080.SR': {'name': 'Arab National Bank', 'sector': 'Banking'},
    '1120.SR': {'name': 'Al Jazira Bank', 'sector': 'Banking'},
    '1140.SR': {'name': 'Bank Albilad', 'sector': 'Banking'},
    '1150.SR': {'name': 'Alinma Bank', 'sector': 'Banking'},
    
    # Energy
    '2222.SR': {'name': 'Saudi Aramco', 'sector': 'Energy'},
    '2030.SR': {'name': 'Saudi Kayan', 'sector': 'Energy'},
    '2310.SR': {'name': 'Sipchem', 'sector': 'Energy'},
    '2380.SR': {'name': 'Petro Rabigh', 'sector': 'Energy'},
    
    # Materials
    '2010.SR': {'name': 'SABIC', 'sector': 'Materials'},
    '1211.SR': {'name': 'Maaden', 'sector': 'Materials'},
    '3020.SR': {'name': 'Yamama Cement', 'sector': 'Materials'},
    '3060.SR': {'name': 'Yanbu Cement', 'sector': 'Materials'},
    '1320.SR': {'name': 'Saudi Steel Pipe', 'sector': 'Materials'},
    '1321.SR': {'name': 'East Pipes', 'sector': 'Materials'},
    '1322.SR': {'name': 'Almasane Mining', 'sector': 'Materials'},
    '2040.SR': {'name': 'Saudi Ceramic', 'sector': 'Materials'},
    
    # Real Estate
    '4300.SR': {'name': 'Dar Al Arkan', 'sector': 'Real Estate'},
    '4310.SR': {'name': 'Emaar Economic City', 'sector': 'Real Estate'},
    '4250.SR': {'name': 'Jabal Omar', 'sector': 'Real Estate'},
    '4100.SR': {'name': 'Makkah Construction', 'sector': 'Real Estate'},
    
    # Retail
    '4190.SR': {'name': 'Jarir Marketing', 'sector': 'Retail'},
    '4001.SR': {'name': 'Al Othaim Markets', 'sector': 'Retail'},
    '4003.SR': {'name': 'Extra', 'sector': 'Retail'},
    
    # Telecom
    '7010.SR': {'name': 'STC', 'sector': 'Telecom'},
    '7020.SR': {'name': 'Mobily', 'sector': 'Telecom'},
    '7030.SR': {'name': 'Zain KSA', 'sector': 'Telecom'},
    
    # Healthcare
    '4002.SR': {'name': 'Mouwasat', 'sector': 'Healthcare'},
    '4004.SR': {'name': 'Dallah Healthcare', 'sector': 'Healthcare'},
    '4009.SR': {'name': 'Middle East Healthcare', 'sector': 'Healthcare'},
    
    # Food & Beverages
    '2050.SR': {'name': 'Savola', 'sector': 'Food'},
    '2280.SR': {'name': 'Almarai', 'sector': 'Food'},
    '6001.SR': {'name': 'Halwani Bros', 'sector': 'Food'},
    '6002.SR': {'name': 'Herfy', 'sector': 'Food'},
    
    # Insurance
    '8010.SR': {'name': 'Tawuniya', 'sector': 'Insurance'},
    '8210.SR': {'name': 'Bupa Arabia', 'sector': 'Insurance'},
    '8050.SR': {'name': 'Salama', 'sector': 'Insurance'},
    
    # Industrial
    '2070.SR': {'name': 'Saudi Pharma', 'sector': 'Industrial'},
    '1303.SR': {'name': 'Electrical Industries', 'sector': 'Industrial'},
    '1304.SR': {'name': 'Yamamah Steel', 'sector': 'Industrial'},
    '2320.SR': {'name': 'Al-Babtain Power', 'sector': 'Industrial'},
    
    # Holding/Diversified
    '4280.SR': {'name': 'Kingdom Holding', 'sector': 'Holding'},
    '2082.SR': {'name': 'ACWA Power', 'sector': 'Utilities'},
}


# ============================================================================
# PORTFOLIO POSITION
# ============================================================================

@dataclass
class Position:
    """Active portfolio position"""
    ticker: str
    name: str
    sector: str
    entry_date: datetime
    entry_price: float
    shares: int
    cost_basis: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    highest_price: float
    signal_probability: float
    signal_confidence: float
    screening_score: float
    

@dataclass 
class ClosedTrade:
    """Record of a closed trade"""
    ticker: str
    name: str
    sector: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str


# ============================================================================
# POINT-IN-TIME SIGNAL GENERATOR
# ============================================================================

class PITSignalGenerator:
    """Point-in-Time signal generator - NO LOOK-AHEAD"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR using only available data"""
        if len(df) < period + 1:
            return df['close'].iloc[-1] * 0.02
        
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        return tr.tail(period).mean()
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(df) < 20:
            return 0.30
        returns = df['close'].pct_change().dropna()
        return returns.tail(20).std() * np.sqrt(252)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        if len(df) < period + 1:
            return 50
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        
        if loss == 0:
            return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_position(self, df: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands"""
        if len(df) < 20:
            return 0.5
        
        close = df['close']
        sma = close.tail(20).mean()
        std = close.tail(20).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        current = close.iloc[-1]
        if upper == lower:
            return 0.5
        return (current - lower) / (upper - lower)
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal using ONLY historical data.
        
        NO FUTURE DATA IS USED.
        """
        if len(df) < 50:
            return {'signal': 0, 'probability': 0, 'confidence': 0}
        
        close = df['close']
        
        # Technical indicators (all calculated from historical data only)
        rsi = self.calculate_rsi(df)
        bb_pos = self.calculate_bollinger_position(df)
        volatility = self.calculate_volatility(df)
        atr = self.calculate_atr(df)
        
        # Moving averages
        sma_20 = close.tail(20).mean()
        sma_50 = close.tail(50).mean()
        current_price = close.iloc[-1]
        
        # Momentum
        momentum_5d = current_price / close.iloc[-6] - 1 if len(close) > 5 else 0
        momentum_20d = current_price / close.iloc[-21] - 1 if len(close) > 20 else 0
        
        # Volume analysis
        if 'volume' in df.columns:
            vol_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
        else:
            vol_ratio = 1.0
        
        # Generate signal score
        signal_score = 0.0
        confidence_factors = []
        
        # Mean reversion signals
        if bb_pos < 0.2:  # Near lower Bollinger Band
            signal_score += 0.25
            confidence_factors.append(0.7)
        
        if rsi < 30:  # Oversold
            signal_score += 0.20
            confidence_factors.append(0.65)
        elif rsi < 40:
            signal_score += 0.10
            confidence_factors.append(0.5)
        
        # Trend confirmation
        if current_price > sma_50:  # Above long-term trend
            signal_score += 0.15
            confidence_factors.append(0.6)
        
        if momentum_5d > 0:  # Short-term momentum positive
            signal_score += 0.15
            confidence_factors.append(0.55)
        
        # Volume confirmation
        if vol_ratio > 1.5:  # Above average volume
            signal_score += 0.15
            confidence_factors.append(0.5)
        
        # Volatility opportunity
        if volatility > 0.25:  # Higher volatility = more opportunity
            signal_score += 0.10
            confidence_factors.append(0.4)
        
        probability = min(signal_score, 1.0)
        confidence = np.mean(confidence_factors) if confidence_factors else 0.3
        
        # Generate binary signal
        signal = 1 if (
            probability >= self.config.min_signal_probability and
            confidence >= self.config.min_signal_confidence
        ) else 0
        
        return {
            'signal': signal,
            'probability': probability,
            'confidence': confidence,
            'rsi': rsi,
            'bb_position': bb_pos,
            'volatility': volatility,
            'atr': atr,
            'momentum_5d': momentum_5d,
            'vol_ratio': vol_ratio
        }


# ============================================================================
# STOCK SCREENER (Point-in-Time)
# ============================================================================

class PITScreener:
    """Point-in-Time stock screener"""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
    
    def screen_stock(self, df: pd.DataFrame, market_df: pd.DataFrame = None) -> Dict:
        """
        Screen a stock using only historical data available at that point.
        """
        if len(df) < 100:
            return {'score': 0, 'qualified': False}
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Momentum score (0-20)
        mom_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 20 else 0
        mom_60d = close.iloc[-1] / close.iloc[-61] - 1 if len(close) > 60 else 0
        momentum_score = min(20, max(0, (mom_20d + mom_60d) * 50 + 10))
        
        # Trend strength score (0-20)
        sma_50 = close.tail(50).mean()
        sma_200 = close.tail(200).mean() if len(close) >= 200 else sma_50
        trend_score = 15 if close.iloc[-1] > sma_50 > sma_200 else 10 if close.iloc[-1] > sma_50 else 5
        
        # Volatility score (0-15) - moderate volatility is good
        volatility = returns.tail(60).std() * np.sqrt(252)
        if 0.20 <= volatility <= 0.40:
            vol_score = 15
        elif 0.15 <= volatility <= 0.50:
            vol_score = 10
        else:
            vol_score = 5
        
        # Liquidity score (0-15)
        if 'volume' in df.columns:
            avg_volume = df['volume'].tail(20).mean()
            avg_value = avg_volume * close.iloc[-1]
            if avg_value > 10_000_000:  # 10M+ SAR daily
                liq_score = 15
            elif avg_value > 1_000_000:
                liq_score = 10
            else:
                liq_score = 5
        else:
            liq_score = 10
        
        # RSI position score (0-15) - oversold is opportunity
        rsi = self._calculate_rsi(df)
        if rsi < 35:
            rsi_score = 15
        elif rsi < 45:
            rsi_score = 10
        elif rsi > 70:
            rsi_score = 5
        else:
            rsi_score = 8
        
        # Price action score (0-15)
        bb_pos = self._calculate_bb_position(df)
        if bb_pos < 0.3:
            pa_score = 15
        elif bb_pos < 0.5:
            pa_score = 10
        else:
            pa_score = 5
        
        total_score = momentum_score + trend_score + vol_score + liq_score + rsi_score + pa_score
        
        return {
            'score': total_score,
            'qualified': total_score >= self.config.min_screening_score,
            'momentum_score': momentum_score,
            'trend_score': trend_score,
            'volatility_score': vol_score,
            'liquidity_score': liq_score,
            'rsi_score': rsi_score,
            'price_action_score': pa_score,
            'volatility': volatility
        }
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 50
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        if loss == 0:
            return 100
        return 100 - (100 / (1 + gain / loss))
    
    def _calculate_bb_position(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.5
        close = df['close']
        sma = close.tail(20).mean()
        std = close.tail(20).std()
        if std == 0:
            return 0.5
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (close.iloc[-1] - lower) / (upper - lower)


# ============================================================================
# PORTFOLIO SIMULATOR
# ============================================================================

class PortfolioSimulator:
    """
    Full portfolio simulation with realistic execution.
    
    Features:
    - Multi-stock portfolio management
    - Position sizing and limits
    - Sector exposure limits
    - Transaction costs
    - Point-in-time decision making
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.signal_gen = PITSignalGenerator(config)
        self.screener = PITScreener(config)
        
        # State
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.equity_history: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        
    def reset(self):
        """Reset simulation state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.daily_snapshots = []
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos.shares * prices.get(ticker, pos.entry_price)
            for ticker, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate current sector exposure"""
        sector_values = {}
        total_positions = sum(pos.shares * pos.entry_price for pos in self.positions.values())
        
        if total_positions == 0:
            return {}
        
        for ticker, pos in self.positions.items():
            sector = pos.sector
            value = pos.shares * pos.entry_price
            sector_values[sector] = sector_values.get(sector, 0) + value / total_positions
        
        return sector_values
    
    def can_add_position(self, sector: str) -> bool:
        """Check if we can add a new position"""
        if len(self.positions) >= self.config.max_positions:
            return False
        
        sector_exposure = self.calculate_sector_exposure()
        if sector_exposure.get(sector, 0) >= self.config.max_sector_exposure:
            return False
        
        return True
    
    def calculate_position_size(
        self, 
        price: float, 
        volatility: float,
        atr: float
    ) -> Tuple[int, float]:
        """
        Calculate position size based on risk management rules.
        
        Returns (shares, stop_loss_price)
        """
        # Method 1: Fixed percentage of capital
        max_position = self.cash * self.config.max_position_pct
        
        # Method 2: Risk-based (max 2% risk per trade)
        stop_distance = atr * self.config.stop_loss_atr_mult
        risk_based_shares = (self.cash * self.config.max_single_stock_risk) / stop_distance
        risk_based_value = risk_based_shares * price
        
        # Method 3: Volatility-adjusted
        vol_factor = min(1.0, 0.25 / max(volatility, 0.01))
        vol_adjusted = max_position * vol_factor
        
        # Take minimum of all methods
        position_value = min(max_position, risk_based_value, vol_adjusted)
        position_value = max(position_value, self.config.min_position_value)
        
        shares = int(position_value / price)
        stop_loss = price - stop_distance
        
        return shares, stop_loss
    
    def process_day(
        self,
        date: datetime,
        stock_data: Dict[str, pd.DataFrame],
        next_day_opens: Dict[str, float]
    ) -> Dict:
        """
        Process a single trading day.
        
        All decisions are made using data up to 'date' only.
        Executions happen at next day's open.
        """
        daily_actions = {
            'date': date,
            'entries': [],
            'exits': [],
            'signals': []
        }
        
        current_prices = {}
        
        # Get current prices for all stocks we're tracking
        for ticker, df in stock_data.items():
            if date in df.index:
                current_prices[ticker] = df.loc[date, 'close']
        
        # 1. CHECK EXITS FOR EXISTING POSITIONS
        positions_to_close = []
        
        for ticker, pos in self.positions.items():
            if ticker not in current_prices:
                continue
            
            current_price = current_prices[ticker]
            df = stock_data[ticker]
            hist_df = df[df.index <= date]
            
            holding_days = (date - pos.entry_date).days
            
            # Update trailing stop
            if current_price > pos.highest_price:
                pos.highest_price = current_price
                new_trailing = current_price * (1 - self.config.trailing_stop_pct)
                pos.trailing_stop = max(pos.trailing_stop, new_trailing)
            
            effective_stop = max(pos.stop_loss, pos.trailing_stop)
            
            exit_reason = None
            
            # Check exit conditions
            if current_price <= effective_stop:
                exit_reason = 'stop_loss'
            elif current_price >= pos.take_profit:
                exit_reason = 'take_profit'
            elif holding_days >= self.config.max_holding_days:
                exit_reason = 'time_exit'
            
            if exit_reason:
                positions_to_close.append((ticker, exit_reason))
        
        # Execute exits at next day open
        for ticker, reason in positions_to_close:
            if ticker not in next_day_opens:
                continue
            
            pos = self.positions[ticker]
            exit_price = next_day_opens[ticker] * (1 - self.config.slippage_pct)
            
            gross_pnl = (exit_price - pos.entry_price) * pos.shares
            commission = exit_price * pos.shares * self.config.commission_pct
            net_pnl = gross_pnl - commission
            pnl_pct = (exit_price / pos.entry_price) - 1
            
            trade = ClosedTrade(
                ticker=ticker,
                name=pos.name,
                sector=pos.sector,
                entry_date=pos.entry_date,
                exit_date=date + timedelta(days=1),
                entry_price=pos.entry_price,
                exit_price=exit_price,
                shares=pos.shares,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                pnl_pct=pnl_pct,
                holding_days=(date - pos.entry_date).days,
                exit_reason=reason
            )
            
            self.closed_trades.append(trade)
            self.cash += exit_price * pos.shares - commission
            del self.positions[ticker]
            
            daily_actions['exits'].append({
                'ticker': ticker,
                'reason': reason,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct
            })
        
        # 2. SCREEN AND GENERATE SIGNALS FOR NEW ENTRIES
        candidates = []
        
        for ticker, info in TASI_UNIVERSE.items():
            if ticker in self.positions:
                continue
            
            if ticker not in stock_data:
                continue
            
            df = stock_data[ticker]
            if date not in df.index:
                continue
            
            # Get historical data up to this date ONLY (no look-ahead)
            hist_df = df[df.index <= date].copy()
            
            if len(hist_df) < 100:
                continue
            
            # Screen the stock (point-in-time)
            screen_result = self.screener.screen_stock(hist_df)
            
            if not screen_result['qualified']:
                continue
            
            # Generate signal (point-in-time)
            signal = self.signal_gen.generate_signal(hist_df)
            
            daily_actions['signals'].append({
                'ticker': ticker,
                'score': screen_result['score'],
                'signal': signal['signal'],
                'probability': signal['probability']
            })
            
            if signal['signal'] == 1:
                candidates.append({
                    'ticker': ticker,
                    'info': info,
                    'score': screen_result['score'],
                    'signal': signal,
                    'screen': screen_result
                })
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. ENTER NEW POSITIONS
        for candidate in candidates:
            ticker = candidate['ticker']
            info = candidate['info']
            signal = candidate['signal']
            screen = candidate['screen']
            
            # Check if we can add this position
            if not self.can_add_position(info['sector']):
                continue
            
            if ticker not in next_day_opens:
                continue
            
            entry_price = next_day_opens[ticker] * (1 + self.config.slippage_pct)
            
            # Calculate position size
            shares, stop_loss = self.calculate_position_size(
                entry_price,
                screen['volatility'],
                signal['atr']
            )
            
            if shares <= 0:
                continue
            
            cost = entry_price * shares
            commission = cost * self.config.commission_pct
            total_cost = cost + commission
            
            if total_cost > self.cash:
                continue
            
            # Create position
            position = Position(
                ticker=ticker,
                name=info['name'],
                sector=info['sector'],
                entry_date=date + timedelta(days=1),
                entry_price=entry_price,
                shares=shares,
                cost_basis=total_cost,
                stop_loss=stop_loss,
                take_profit=entry_price * (1 + self.config.take_profit_pct),
                trailing_stop=stop_loss,
                highest_price=entry_price,
                signal_probability=signal['probability'],
                signal_confidence=signal['confidence'],
                screening_score=candidate['score']
            )
            
            self.positions[ticker] = position
            self.cash -= total_cost
            
            daily_actions['entries'].append({
                'ticker': ticker,
                'shares': shares,
                'price': entry_price,
                'stop_loss': stop_loss
            })
        
        # 4. RECORD PORTFOLIO SNAPSHOT
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        snapshot = {
            'date': date,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_value': portfolio_value,
            'num_positions': len(self.positions),
            'daily_return': 0  # Will be calculated later
        }
        
        if self.equity_history:
            prev_value = self.equity_history[-1]['total_value']
            snapshot['daily_return'] = (portfolio_value / prev_value) - 1
        
        self.equity_history.append(snapshot)
        self.daily_snapshots.append(daily_actions)
        
        return daily_actions
    
    def run_simulation(
        self,
        start_date: str = "2022-01-01",
        end_date: str = None
    ) -> Dict:
        """
        Run the full portfolio simulation.
        """
        print("=" * 80)
        print("FULL PORTFOLIO SIMULATION - TASI MARKET")
        print("=" * 80)
        print(f"Initial Capital: {self.config.initial_capital:,.0f} SAR")
        print(f"Start Date: {start_date}")
        print(f"Max Positions: {self.config.max_positions}")
        print("=" * 80)
        
        self.reset()
        
        # 1. FETCH ALL STOCK DATA
        print("\n[1] FETCHING STOCK DATA...")
        stock_data = {}
        failed_tickers = []
        
        for ticker in TASI_UNIVERSE.keys():
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if len(df) < 100:
                    failed_tickers.append(ticker)
                    continue
                
                df.columns = [c.lower() for c in df.columns]
                stock_data[ticker] = df
                
            except Exception as e:
                failed_tickers.append(ticker)
        
        print(f"  Loaded: {len(stock_data)} stocks")
        print(f"  Failed: {len(failed_tickers)} stocks")
        
        # Get common trading dates
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)
        
        print(f"  Trading Days: {len(trading_dates)}")
        print(f"  Date Range: {trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}")
        
        # 2. RUN SIMULATION DAY BY DAY
        print("\n[2] RUNNING SIMULATION...")
        
        for i, date in enumerate(trading_dates[:-1]):  # -1 for next-day execution
            next_date = trading_dates[i + 1]
            
            # Get next day opens for execution
            next_day_opens = {}
            for ticker, df in stock_data.items():
                if next_date in df.index:
                    if 'open' in df.columns:
                        next_day_opens[ticker] = df.loc[next_date, 'open']
                    else:
                        next_day_opens[ticker] = df.loc[next_date, 'close']
            
            # Process the day
            actions = self.process_day(date, stock_data, next_day_opens)
            
            # Progress update
            if (i + 1) % 50 == 0 or i == 0:
                current_value = self.equity_history[-1]['total_value']
                pnl = current_value - self.config.initial_capital
                pnl_pct = (current_value / self.config.initial_capital - 1) * 100
                print(f"  {date.strftime('%Y-%m-%d')}: Value={current_value:,.0f} SAR, "
                      f"P&L={pnl:+,.0f} ({pnl_pct:+.1f}%), Positions={len(self.positions)}")
        
        # 3. CLOSE ANY REMAINING POSITIONS
        print("\n[3] CLOSING REMAINING POSITIONS...")
        final_date = trading_dates[-1]
        
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            
            if ticker in stock_data and final_date in stock_data[ticker].index:
                exit_price = stock_data[ticker].loc[final_date, 'close']
            else:
                exit_price = pos.entry_price
            
            gross_pnl = (exit_price - pos.entry_price) * pos.shares
            commission = exit_price * pos.shares * self.config.commission_pct
            net_pnl = gross_pnl - commission
            
            trade = ClosedTrade(
                ticker=ticker,
                name=pos.name,
                sector=pos.sector,
                entry_date=pos.entry_date,
                exit_date=final_date,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                shares=pos.shares,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                pnl_pct=(exit_price / pos.entry_price) - 1,
                holding_days=(final_date - pos.entry_date).days,
                exit_reason='simulation_end'
            )
            
            self.closed_trades.append(trade)
            self.cash += exit_price * pos.shares - commission
        
        self.positions = {}
        
        # 4. CALCULATE RESULTS
        results = self.calculate_results()
        
        return results
    
    def calculate_results(self) -> Dict:
        """Calculate comprehensive simulation results"""
        
        if not self.equity_history:
            return {}
        
        equity = pd.DataFrame(self.equity_history)
        equity['date'] = pd.to_datetime(equity['date'])
        equity = equity.set_index('date')
        
        trades_df = pd.DataFrame([{
            'ticker': t.ticker,
            'name': t.name,
            'sector': t.sector,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'gross_pnl': t.gross_pnl,
            'net_pnl': t.net_pnl,
            'pnl_pct': t.pnl_pct,
            'holding_days': t.holding_days,
            'exit_reason': t.exit_reason
        } for t in self.closed_trades])
        
        # Basic metrics
        initial_capital = self.config.initial_capital
        final_value = self.cash
        total_return = (final_value / initial_capital) - 1
        
        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.net_pnl > 0])
        losing_trades = len([t for t in self.closed_trades if t.net_pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.net_pnl for t in self.closed_trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.closed_trades if t.net_pnl <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Time-based returns
        days = len(equity)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        daily_returns = equity['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        negative_returns = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        
        # Drawdown
        equity_values = equity['total_value'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = drawdown.max()
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # By period analysis
        equity['month'] = equity.index.to_period('M')
        monthly_returns = equity.groupby('month')['total_value'].last().pct_change()
        
        results = {
            'summary': {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_pnl': final_value - initial_capital,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar,
                'trading_days': days,
                'years': years
            },
            'trades': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_holding_days': np.mean([t.holding_days for t in self.closed_trades]) if self.closed_trades else 0,
                'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss
            },
            'equity': equity,
            'trades_df': trades_df if len(trades_df) > 0 else pd.DataFrame(),
            'monthly_returns': monthly_returns
        }
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_simulation():
    """Run the complete portfolio simulation"""
    
    config = PortfolioConfig(
        initial_capital=1_000_000,
        max_position_pct=0.05,
        max_positions=20,
        min_screening_score=55,
        min_signal_probability=0.55,
        min_signal_confidence=0.35,
        stop_loss_atr_mult=2.0,
        take_profit_pct=0.10,
        trailing_stop_pct=0.03,
        max_holding_days=20
    )
    
    simulator = PortfolioSimulator(config)
    
    # Run simulation from 2022 to present
    results = simulator.run_simulation(start_date="2022-01-01")
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    
    summary = results['summary']
    trades = results['trades']
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                         PORTFOLIO PERFORMANCE SUMMARY                          │
├────────────────────────────────────────────────────────────────────────────────┤
│  Initial Capital:      {summary['initial_capital']:>15,.0f} SAR                           │
│  Final Value:          {summary['final_value']:>15,.0f} SAR                           │
│  Total P&L:            {summary['total_pnl']:>+15,.0f} SAR                           │
│  Total Return:         {summary['total_return']:>+14.1%}                                   │
│  Annual Return:        {summary['annual_return']:>+14.1%}                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                              RISK METRICS                                      │
├────────────────────────────────────────────────────────────────────────────────┤
│  Sharpe Ratio:         {summary['sharpe_ratio']:>15.2f}                                    │
│  Sortino Ratio:        {summary['sortino_ratio']:>15.2f}                                    │
│  Max Drawdown:         {summary['max_drawdown']:>14.1%}                                    │
│  Calmar Ratio:         {summary['calmar_ratio']:>15.2f}                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                             TRADE STATISTICS                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│  Total Trades:         {trades['total_trades']:>15}                                    │
│  Winning Trades:       {trades['winning_trades']:>15}                                    │
│  Losing Trades:        {trades['losing_trades']:>15}                                    │
│  Win Rate:             {trades['win_rate']:>14.1%}                                    │
│  Profit Factor:        {trades['profit_factor']:>15.2f}                                    │
│  Avg Win:              {trades['avg_win']:>+15,.0f} SAR                           │
│  Avg Loss:             {trades['avg_loss']:>+15,.0f} SAR                           │
│  Avg Holding Days:     {trades['avg_holding_days']:>15.1f}                                    │
│  Expectancy:           {trades['expectancy']:>+15,.0f} SAR/trade                     │
└────────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Trade analysis by exit reason
    if len(results['trades_df']) > 0:
        trades_df = results['trades_df']
        
        print("\n" + "-" * 80)
        print("TRADES BY EXIT REASON")
        print("-" * 80)
        
        exit_analysis = trades_df.groupby('exit_reason').agg({
            'ticker': 'count',
            'net_pnl': ['sum', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)
        
        exit_analysis.columns = ['Count', 'Total P&L', 'Avg P&L', 'Avg %']
        print(exit_analysis.to_string())
        
        print("\n" + "-" * 80)
        print("TRADES BY SECTOR")
        print("-" * 80)
        
        sector_analysis = trades_df.groupby('sector').agg({
            'ticker': 'count',
            'net_pnl': ['sum', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)
        
        sector_analysis.columns = ['Count', 'Total P&L', 'Avg P&L', 'Avg %']
        print(sector_analysis.to_string())
        
        print("\n" + "-" * 80)
        print("TOP 10 BEST TRADES")
        print("-" * 80)
        
        best_trades = trades_df.nlargest(10, 'net_pnl')[['ticker', 'name', 'entry_date', 'exit_date', 'pnl_pct', 'net_pnl', 'exit_reason']]
        best_trades['entry_date'] = pd.to_datetime(best_trades['entry_date']).dt.strftime('%Y-%m-%d')
        best_trades['exit_date'] = pd.to_datetime(best_trades['exit_date']).dt.strftime('%Y-%m-%d')
        best_trades['pnl_pct'] = best_trades['pnl_pct'].apply(lambda x: f"{x:+.1%}")
        best_trades['net_pnl'] = best_trades['net_pnl'].apply(lambda x: f"{x:+,.0f}")
        print(best_trades.to_string(index=False))
        
        print("\n" + "-" * 80)
        print("TOP 10 WORST TRADES")
        print("-" * 80)
        
        worst_trades = trades_df.nsmallest(10, 'net_pnl')[['ticker', 'name', 'entry_date', 'exit_date', 'pnl_pct', 'net_pnl', 'exit_reason']]
        worst_trades['entry_date'] = pd.to_datetime(worst_trades['entry_date']).dt.strftime('%Y-%m-%d')
        worst_trades['exit_date'] = pd.to_datetime(worst_trades['exit_date']).dt.strftime('%Y-%m-%d')
        worst_trades['pnl_pct'] = worst_trades['pnl_pct'].apply(lambda x: f"{x:+.1%}")
        worst_trades['net_pnl'] = worst_trades['net_pnl'].apply(lambda x: f"{x:+,.0f}")
        print(worst_trades.to_string(index=False))
    
    # Monthly returns
    if 'monthly_returns' in results and len(results['monthly_returns']) > 0:
        print("\n" + "-" * 80)
        print("MONTHLY RETURNS")
        print("-" * 80)
        
        monthly = results['monthly_returns'].dropna()
        for period, ret in monthly.items():
            bar = "█" * int(abs(ret) * 100) if abs(ret) < 0.5 else "█" * 50
            color_indicator = "+" if ret >= 0 else "-"
            print(f"  {period}: {ret:+.2%} {color_indicator}{bar}")
    
    # Save results
    os.makedirs('output/portfolio_simulation', exist_ok=True)
    
    # Save equity curve
    results['equity'].to_csv('output/portfolio_simulation/equity_curve.csv')
    
    # Save trades
    if len(results['trades_df']) > 0:
        results['trades_df'].to_csv('output/portfolio_simulation/all_trades.csv', index=False)
    
    # Save summary
    with open('output/portfolio_simulation/simulation_summary.json', 'w') as f:
        json.dump({
            'summary': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in summary.items()},
            'trades': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in trades.items()}
        }, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print("  output/portfolio_simulation/equity_curve.csv")
    print("  output/portfolio_simulation/all_trades.csv")
    print("  output/portfolio_simulation/simulation_summary.json")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("STRATEGY VERDICT")
    print("=" * 80)
    
    checks = []
    
    if summary['total_return'] > 0:
        checks.append(("✓", "Positive total return"))
    else:
        checks.append(("✗", "Negative total return"))
    
    if summary['sharpe_ratio'] > 0.5:
        checks.append(("✓", f"Good risk-adjusted return (Sharpe={summary['sharpe_ratio']:.2f})"))
    elif summary['sharpe_ratio'] > 0:
        checks.append(("⚠️", f"Marginal risk-adjusted return (Sharpe={summary['sharpe_ratio']:.2f})"))
    else:
        checks.append(("✗", f"Poor risk-adjusted return (Sharpe={summary['sharpe_ratio']:.2f})"))
    
    if summary['max_drawdown'] < 0.15:
        checks.append(("✓", f"Drawdown within limits ({summary['max_drawdown']:.1%})"))
    elif summary['max_drawdown'] < 0.25:
        checks.append(("⚠️", f"Moderate drawdown ({summary['max_drawdown']:.1%})"))
    else:
        checks.append(("✗", f"Excessive drawdown ({summary['max_drawdown']:.1%})"))
    
    if trades['win_rate'] > 0.45:
        checks.append(("✓", f"Acceptable win rate ({trades['win_rate']:.1%})"))
    else:
        checks.append(("⚠️", f"Low win rate ({trades['win_rate']:.1%})"))
    
    if trades['profit_factor'] > 1.0:
        checks.append(("✓", f"Profitable trading (PF={trades['profit_factor']:.2f})"))
    else:
        checks.append(("✗", f"Unprofitable trading (PF={trades['profit_factor']:.2f})"))
    
    if trades['total_trades'] >= 50:
        checks.append(("✓", f"Sufficient trades for validation ({trades['total_trades']})"))
    else:
        checks.append(("⚠️", f"Limited trades ({trades['total_trades']})"))
    
    for symbol, message in checks:
        print(f"  {symbol} {message}")
    
    passed = sum(1 for s, _ in checks if s == "✓")
    warnings = sum(1 for s, _ in checks if s == "⚠️")
    
    print(f"\n  Score: {passed}/{len(checks)} passed, {warnings} warnings")
    
    if passed >= 5:
        print("\n  🏆 STRATEGY VALIDATED - Ready for live paper trading")
    elif passed >= 3:
        print("\n  ⚠️  STRATEGY NEEDS REFINEMENT - Some concerns to address")
    else:
        print("\n  ❌ STRATEGY NOT VALIDATED - Major issues detected")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_full_simulation()
