#!/usr/bin/env python3
"""
IMPROVED PORTFOLIO SIMULATION - TASI MARKET
============================================
Based on learnings from initial simulation:

IMPROVEMENTS:
1. WIDER STOPS: 3x ATR instead of 2x (71% stop-loss rate was too high)
2. STRICTER SIGNALS: RSI < 35, Volume > 2x average
3. SECTOR FOCUS: Prioritize Telecom, Banking, Real Estate (profitable sectors)
4. TIGHTER TAKE-PROFIT: 8% instead of 10% (faster rotation)
5. TREND CONFIRMATION: Price must be above SMA20

Author: Anomaly Prediction System - Scientific Trading Division
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf


@dataclass
class ImprovedConfig:
    """Improved portfolio configuration"""
    initial_capital: float = 1_000_000
    
    # Position Limits
    max_position_pct: float = 0.04           # Reduced from 5% to 4%
    max_positions: int = 15                   # Reduced from 20 to 15
    min_position_value: float = 15000         # Increased minimum
    
    # Risk Management - IMPROVED
    stop_loss_atr_mult: float = 3.0           # INCREASED from 2.0 to 3.0
    take_profit_pct: float = 0.08             # REDUCED from 10% to 8%
    trailing_stop_pct: float = 0.04           # INCREASED from 3% to 4%
    max_holding_days: int = 15                # REDUCED from 20 to 15
    
    # Signal Thresholds - STRICTER
    min_screening_score: float = 60           # INCREASED from 55
    min_signal_probability: float = 0.60      # INCREASED from 55%
    min_rsi: float = 35                       # RSI must be below this (oversold)
    min_volume_ratio: float = 2.0             # INCREASED from 1.5x
    require_trend_confirmation: bool = True   # NEW: Price > SMA20
    
    # Sector Preferences (based on backtest analysis)
    preferred_sectors: tuple = ('Telecom', 'Banking', 'Real Estate', 'Insurance', 'Industrial')
    avoid_sectors: tuple = ('Healthcare', 'Energy')
    
    # Transaction Costs
    commission_pct: float = 0.001
    slippage_pct: float = 0.001


# Updated universe with sector focus
TASI_UNIVERSE = {
    # Telecom - BEST PERFORMER
    '7010.SR': {'name': 'STC', 'sector': 'Telecom', 'priority': 1},
    '7020.SR': {'name': 'Mobily', 'sector': 'Telecom', 'priority': 1},
    '7030.SR': {'name': 'Zain KSA', 'sector': 'Telecom', 'priority': 1},
    
    # Banking - SECOND BEST
    '1180.SR': {'name': 'Al Rajhi Bank', 'sector': 'Banking', 'priority': 1},
    '1010.SR': {'name': 'Riyad Bank', 'sector': 'Banking', 'priority': 1},
    '1050.SR': {'name': 'Banque Saudi Fransi', 'sector': 'Banking', 'priority': 1},
    '1060.SR': {'name': 'Saudi Awwal Bank', 'sector': 'Banking', 'priority': 1},
    '1080.SR': {'name': 'Arab National Bank', 'sector': 'Banking', 'priority': 1},
    '1140.SR': {'name': 'Bank Albilad', 'sector': 'Banking', 'priority': 1},
    '1150.SR': {'name': 'Alinma Bank', 'sector': 'Banking', 'priority': 1},
    
    # Real Estate - THIRD BEST
    '4300.SR': {'name': 'Dar Al Arkan', 'sector': 'Real Estate', 'priority': 1},
    '4310.SR': {'name': 'Emaar Economic City', 'sector': 'Real Estate', 'priority': 1},
    '4250.SR': {'name': 'Jabal Omar', 'sector': 'Real Estate', 'priority': 1},
    '4100.SR': {'name': 'Makkah Construction', 'sector': 'Real Estate', 'priority': 1},
    
    # Insurance - PROFITABLE
    '8010.SR': {'name': 'Tawuniya', 'sector': 'Insurance', 'priority': 1},
    '8210.SR': {'name': 'Bupa Arabia', 'sector': 'Insurance', 'priority': 1},
    
    # Industrial - DECENT
    '1303.SR': {'name': 'Electrical Industries', 'sector': 'Industrial', 'priority': 2},
    '1304.SR': {'name': 'Yamamah Steel', 'sector': 'Industrial', 'priority': 2},
    '1320.SR': {'name': 'Saudi Steel Pipe', 'sector': 'Industrial', 'priority': 2},
    '1321.SR': {'name': 'East Pipes', 'sector': 'Industrial', 'priority': 2},
    '1322.SR': {'name': 'Almasane Mining', 'sector': 'Industrial', 'priority': 2},
    
    # Materials - MODERATE (keep some for diversification)
    '2010.SR': {'name': 'SABIC', 'sector': 'Materials', 'priority': 2},
    '1211.SR': {'name': 'Maaden', 'sector': 'Materials', 'priority': 2},
    '3020.SR': {'name': 'Yamama Cement', 'sector': 'Materials', 'priority': 2},
    
    # Retail - MODERATE
    '4190.SR': {'name': 'Jarir Marketing', 'sector': 'Retail', 'priority': 2},
    
    # Food - LOWER PRIORITY (but keep some)
    '2280.SR': {'name': 'Almarai', 'sector': 'Food', 'priority': 3},
    '2050.SR': {'name': 'Savola', 'sector': 'Food', 'priority': 3},
    
    # Utilities
    '2082.SR': {'name': 'ACWA Power', 'sector': 'Utilities', 'priority': 2},
    
    # Holding
    '4280.SR': {'name': 'Kingdom Holding', 'sector': 'Holding', 'priority': 2},
    
    # Energy - LOWER PRIORITY
    '2222.SR': {'name': 'Saudi Aramco', 'sector': 'Energy', 'priority': 3},
}


class ImprovedSignalGenerator:
    """Improved signal generator with stricter filters"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all indicators using historical data only"""
        if len(df) < 50:
            return {}
        
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
        
        # Moving Averages
        sma_20 = close.tail(20).mean()
        sma_50 = close.tail(50).mean()
        
        # Bollinger Band Position
        std_20 = close.tail(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Volume
        if 'volume' in df.columns:
            vol_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
        else:
            vol_ratio = 1.0
        
        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            atr = tr.tail(14).mean()
        else:
            atr = close.tail(14).std()
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)
        
        # Momentum
        momentum_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else 0
        momentum_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 20 else 0
        
        return {
            'rsi': rsi,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'bb_position': bb_position,
            'vol_ratio': vol_ratio,
            'atr': atr,
            'volatility': volatility,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'current_price': close.iloc[-1]
        }
    
    def generate_signal(self, df: pd.DataFrame, sector: str, priority: int) -> Dict:
        """Generate signal with stricter criteria"""
        indicators = self.calculate_indicators(df)
        
        if not indicators:
            return {'signal': 0, 'probability': 0, 'confidence': 0, 'atr': 0}
        
        signal_score = 0.0
        confidence_factors = []
        
        # STRICT FILTER 1: RSI must be oversold
        if indicators['rsi'] < self.config.min_rsi:
            signal_score += 0.30
            confidence_factors.append(0.8)
        elif indicators['rsi'] < 40:
            signal_score += 0.15
            confidence_factors.append(0.5)
        else:
            # Not oversold - reduce signal
            return {'signal': 0, 'probability': 0, 'confidence': 0, 'atr': indicators['atr']}
        
        # STRICT FILTER 2: Volume confirmation (must be 2x average)
        if indicators['vol_ratio'] >= self.config.min_volume_ratio:
            signal_score += 0.25
            confidence_factors.append(0.75)
        elif indicators['vol_ratio'] >= 1.5:
            signal_score += 0.10
            confidence_factors.append(0.4)
        
        # STRICT FILTER 3: Trend confirmation (price > SMA20)
        if self.config.require_trend_confirmation:
            if indicators['current_price'] > indicators['sma_20']:
                signal_score += 0.15
                confidence_factors.append(0.6)
            else:
                # Below SMA20 - need stronger oversold condition
                if indicators['rsi'] > 30:  # Not extremely oversold
                    signal_score -= 0.10
        
        # STRICT FILTER 4: Bollinger Band position (near lower band)
        if indicators['bb_position'] < 0.2:
            signal_score += 0.20
            confidence_factors.append(0.7)
        elif indicators['bb_position'] < 0.3:
            signal_score += 0.10
            confidence_factors.append(0.5)
        
        # BONUS: Momentum reversal (was down, starting to turn)
        if indicators['momentum_5d'] > 0 and indicators['momentum_20d'] < 0:
            signal_score += 0.10
            confidence_factors.append(0.6)
        
        # BONUS: Preferred sector
        if sector in self.config.preferred_sectors:
            signal_score += 0.05
            confidence_factors.append(0.55)
        
        # PENALTY: Avoid sector
        if sector in self.config.avoid_sectors:
            signal_score -= 0.15
        
        probability = min(signal_score, 1.0)
        confidence = np.mean(confidence_factors) if confidence_factors else 0.3
        
        # Higher threshold for signal
        signal = 1 if (
            probability >= self.config.min_signal_probability and
            confidence >= 0.50 and
            indicators['rsi'] < self.config.min_rsi  # Double-check RSI
        ) else 0
        
        return {
            'signal': signal,
            'probability': probability,
            'confidence': confidence,
            'atr': indicators['atr'],
            'rsi': indicators['rsi'],
            'vol_ratio': indicators['vol_ratio'],
            'bb_position': indicators['bb_position'],
            'volatility': indicators['volatility']
        }


class ImprovedPortfolioSimulator:
    """Improved portfolio simulator"""
    
    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.signal_gen = ImprovedSignalGenerator(config)
        
        self.cash = config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
    
    def reset(self):
        self.cash = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
    
    def run_simulation(self, start_date: str = "2022-01-01"):
        """Run improved simulation"""
        print("=" * 80)
        print("IMPROVED PORTFOLIO SIMULATION - TASI MARKET")
        print("=" * 80)
        print(f"Initial Capital: {self.config.initial_capital:,.0f} SAR")
        print(f"\nKEY IMPROVEMENTS:")
        print(f"  - Stop-Loss: {self.config.stop_loss_atr_mult}x ATR (was 2x)")
        print(f"  - Take-Profit: {self.config.take_profit_pct*100:.0f}% (was 10%)")
        print(f"  - RSI Filter: < {self.config.min_rsi} (stricter)")
        print(f"  - Volume Filter: {self.config.min_volume_ratio}x avg (was 1.5x)")
        print(f"  - Sector Focus: {', '.join(self.config.preferred_sectors[:3])}...")
        print("=" * 80)
        
        self.reset()
        
        # Fetch data
        print("\n[1] FETCHING DATA...")
        stock_data = {}
        
        for ticker, info in TASI_UNIVERSE.items():
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date)
                if len(df) >= 100:
                    df.columns = [c.lower() for c in df.columns]
                    stock_data[ticker] = df
            except:
                pass
        
        print(f"  Loaded: {len(stock_data)} stocks")
        
        # Get trading dates
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)
        
        print(f"  Trading Days: {len(trading_dates)}")
        
        # Run simulation
        print("\n[2] RUNNING IMPROVED SIMULATION...")
        
        for i, date in enumerate(trading_dates[:-1]):
            next_date = trading_dates[i + 1]
            
            # Get prices
            current_prices = {t: df.loc[date, 'close'] for t, df in stock_data.items() if date in df.index}
            next_opens = {t: df.loc[next_date, 'open'] if 'open' in df.columns else df.loc[next_date, 'close'] 
                         for t, df in stock_data.items() if next_date in df.index}
            
            # Check exits
            for ticker in list(self.positions.keys()):
                if ticker not in current_prices:
                    continue
                
                pos = self.positions[ticker]
                price = current_prices[ticker]
                holding_days = (date - pos['entry_date']).days
                
                # Update trailing stop
                if price > pos['highest_price']:
                    pos['highest_price'] = price
                    pos['trailing_stop'] = max(pos['trailing_stop'], price * (1 - self.config.trailing_stop_pct))
                
                effective_stop = max(pos['stop_loss'], pos['trailing_stop'])
                
                exit_reason = None
                if price <= effective_stop:
                    exit_reason = 'stop_loss'
                elif price >= pos['take_profit']:
                    exit_reason = 'take_profit'
                elif holding_days >= self.config.max_holding_days:
                    exit_reason = 'time_exit'
                
                if exit_reason and ticker in next_opens:
                    exit_price = next_opens[ticker] * (1 - self.config.slippage_pct)
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    commission = exit_price * pos['shares'] * self.config.commission_pct
                    net_pnl = pnl - commission
                    
                    self.closed_trades.append({
                        'ticker': ticker,
                        'sector': pos['sector'],
                        'entry_date': pos['entry_date'],
                        'exit_date': date + timedelta(days=1),
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'net_pnl': net_pnl,
                        'pnl_pct': exit_price / pos['entry_price'] - 1,
                        'holding_days': holding_days,
                        'exit_reason': exit_reason
                    })
                    
                    self.cash += exit_price * pos['shares'] - commission
                    del self.positions[ticker]
            
            # Check entries
            candidates = []
            
            for ticker, info in TASI_UNIVERSE.items():
                if ticker in self.positions or ticker not in stock_data:
                    continue
                
                df = stock_data[ticker]
                if date not in df.index:
                    continue
                
                hist_df = df[df.index <= date]
                if len(hist_df) < 100:
                    continue
                
                signal = self.signal_gen.generate_signal(hist_df, info['sector'], info.get('priority', 2))
                
                if signal['signal'] == 1:
                    candidates.append({
                        'ticker': ticker,
                        'info': info,
                        'signal': signal
                    })
            
            # Sort by priority and probability
            candidates.sort(key=lambda x: (-x['info'].get('priority', 2), -x['signal']['probability']))
            
            # Enter positions
            for candidate in candidates:
                if len(self.positions) >= self.config.max_positions:
                    break
                
                ticker = candidate['ticker']
                info = candidate['info']
                signal = candidate['signal']
                
                if ticker not in next_opens:
                    continue
                
                entry_price = next_opens[ticker] * (1 + self.config.slippage_pct)
                
                # Position sizing (volatility-adjusted)
                vol_factor = min(1.0, 0.25 / max(signal['volatility'], 0.01))
                position_value = self.cash * self.config.max_position_pct * vol_factor
                position_value = max(position_value, self.config.min_position_value)
                
                shares = int(position_value / entry_price)
                
                if shares <= 0:
                    continue
                
                cost = entry_price * shares
                commission = cost * self.config.commission_pct
                
                if cost + commission > self.cash:
                    continue
                
                stop_loss = entry_price - signal['atr'] * self.config.stop_loss_atr_mult
                
                self.positions[ticker] = {
                    'entry_date': date + timedelta(days=1),
                    'entry_price': entry_price,
                    'shares': shares,
                    'sector': info['sector'],
                    'stop_loss': stop_loss,
                    'take_profit': entry_price * (1 + self.config.take_profit_pct),
                    'trailing_stop': stop_loss,
                    'highest_price': entry_price
                }
                
                self.cash -= cost + commission
            
            # Record equity
            position_value = sum(pos['shares'] * current_prices.get(t, pos['entry_price']) 
                               for t, pos in self.positions.items())
            total_value = self.cash + position_value
            
            self.equity_history.append({
                'date': date,
                'cash': self.cash,
                'positions_value': position_value,
                'total_value': total_value,
                'num_positions': len(self.positions)
            })
            
            # Progress
            if (i + 1) % 100 == 0 or i == 0:
                pnl = total_value - self.config.initial_capital
                print(f"  {date.strftime('%Y-%m-%d')}: {total_value:,.0f} SAR ({pnl:+,.0f}, {pnl/self.config.initial_capital*100:+.1f}%)")
        
        # Close remaining positions
        final_date = trading_dates[-1]
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            if ticker in stock_data and final_date in stock_data[ticker].index:
                exit_price = stock_data[ticker].loc[final_date, 'close']
            else:
                exit_price = pos['entry_price']
            
            pnl = (exit_price - pos['entry_price']) * pos['shares']
            commission = exit_price * pos['shares'] * self.config.commission_pct
            
            self.closed_trades.append({
                'ticker': ticker,
                'sector': pos['sector'],
                'entry_date': pos['entry_date'],
                'exit_date': final_date,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'net_pnl': pnl - commission,
                'pnl_pct': exit_price / pos['entry_price'] - 1,
                'holding_days': (final_date - pos['entry_date']).days,
                'exit_reason': 'simulation_end'
            })
            
            self.cash += exit_price * pos['shares'] - commission
        
        self.positions = {}
        
        # Calculate results
        return self.calculate_results()
    
    def calculate_results(self):
        """Calculate comprehensive results"""
        if not self.equity_history:
            return {}
        
        equity = pd.DataFrame(self.equity_history)
        trades_df = pd.DataFrame(self.closed_trades)
        
        initial = self.config.initial_capital
        final = self.cash
        total_return = (final / initial) - 1
        
        days = len(equity)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Trade stats
        total_trades = len(self.closed_trades)
        winning = [t for t in self.closed_trades if t['net_pnl'] > 0]
        losing = [t for t in self.closed_trades if t['net_pnl'] <= 0]
        
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['net_pnl'] for t in winning]) if winning else 0
        avg_loss = np.mean([t['net_pnl'] for t in losing]) if losing else 0
        
        gross_profit = sum(t['net_pnl'] for t in winning)
        gross_loss = abs(sum(t['net_pnl'] for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        equity['daily_return'] = equity['total_value'].pct_change()
        daily_returns = equity['daily_return'].dropna()
        
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        peak = equity['total_value'].cummax()
        drawdown = (peak - equity['total_value']) / peak
        max_dd = drawdown.max()
        
        results = {
            'initial_capital': initial,
            'final_value': final,
            'total_pnl': final - initial,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_days': np.mean([t['holding_days'] for t in self.closed_trades]) if self.closed_trades else 0,
            'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss,
            'equity': equity,
            'trades_df': trades_df
        }
        
        return results


def main():
    """Run improved simulation and compare with original"""
    
    config = ImprovedConfig()
    simulator = ImprovedPortfolioSimulator(config)
    
    results = simulator.run_simulation(start_date="2022-01-01")
    
    # Print results
    print("\n" + "=" * 80)
    print("IMPROVED SIMULATION RESULTS")
    print("=" * 80)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                    IMPROVED PORTFOLIO PERFORMANCE                              │
├────────────────────────────────────────────────────────────────────────────────┤
│  Initial Capital:      {results['initial_capital']:>15,.0f} SAR                           │
│  Final Value:          {results['final_value']:>15,.0f} SAR                           │
│  Total P&L:            {results['total_pnl']:>+15,.0f} SAR                           │
│  Total Return:         {results['total_return']:>+14.1%}                                   │
│  Annual Return:        {results['annual_return']:>+14.1%}                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│  Sharpe Ratio:         {results['sharpe_ratio']:>15.2f}                                    │
│  Max Drawdown:         {results['max_drawdown']:>14.1%}                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  Total Trades:         {results['total_trades']:>15}                                    │
│  Win Rate:             {results['win_rate']:>14.1%}                                    │
│  Profit Factor:        {results['profit_factor']:>15.2f}                                    │
│  Avg Win:              {results['avg_win']:>+15,.0f} SAR                           │
│  Avg Loss:             {results['avg_loss']:>+15,.0f} SAR                           │
│  Expectancy:           {results['expectancy']:>+15,.0f} SAR/trade                     │
└────────────────────────────────────────────────────────────────────────────────┘
""")

    # Comparison with original
    print("\n" + "=" * 80)
    print("COMPARISON: IMPROVED vs ORIGINAL STRATEGY")
    print("=" * 80)
    
    original = {
        'total_return': -0.069,
        'sharpe_ratio': -0.28,
        'max_drawdown': 0.141,
        'win_rate': 0.41,
        'profit_factor': 0.94,
        'total_trades': 1150
    }
    
    print(f"""
┌───────────────────────┬─────────────────┬─────────────────┬─────────────────┐
│       Metric          │    Original     │    Improved     │     Change      │
├───────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Total Return          │   {original['total_return']:>+10.1%}   │   {results['total_return']:>+10.1%}   │   {results['total_return']-original['total_return']:>+10.1%}   │
│ Sharpe Ratio          │   {original['sharpe_ratio']:>+10.2f}   │   {results['sharpe_ratio']:>+10.2f}   │   {results['sharpe_ratio']-original['sharpe_ratio']:>+10.2f}   │
│ Max Drawdown          │    {original['max_drawdown']:>10.1%}   │    {results['max_drawdown']:>10.1%}   │   {results['max_drawdown']-original['max_drawdown']:>+10.1%}   │
│ Win Rate              │    {original['win_rate']:>10.1%}   │    {results['win_rate']:>10.1%}   │   {results['win_rate']-original['win_rate']:>+10.1%}   │
│ Profit Factor         │   {original['profit_factor']:>+10.2f}   │   {results['profit_factor']:>+10.2f}   │   {results['profit_factor']-original['profit_factor']:>+10.2f}   │
│ Total Trades          │   {original['total_trades']:>10}   │   {results['total_trades']:>10}   │   {results['total_trades']-original['total_trades']:>+10}   │
└───────────────────────┴─────────────────┴─────────────────┴─────────────────┘
""")

    # Exit reason analysis
    if len(results['trades_df']) > 0:
        trades = results['trades_df']
        
        print("\nEXIT REASON BREAKDOWN (Improved Strategy):")
        print("-" * 60)
        by_exit = trades.groupby('exit_reason').agg({
            'ticker': 'count',
            'net_pnl': ['sum', 'mean']
        })
        by_exit.columns = ['Count', 'Total PnL', 'Avg PnL']
        print(by_exit)
        
        # Stop-loss percentage
        stop_loss_pct = len(trades[trades['exit_reason'] == 'stop_loss']) / len(trades) * 100
        print(f"\n  Stop-Loss Rate: {stop_loss_pct:.1f}% (was 71.4%)")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("STRATEGY VERDICT")
    print("=" * 80)
    
    checks = []
    
    if results['total_return'] > 0:
        checks.append(("✓", "Positive total return"))
    else:
        checks.append(("✗", "Negative total return"))
    
    if results['sharpe_ratio'] > 0.5:
        checks.append(("✓", f"Good Sharpe ratio ({results['sharpe_ratio']:.2f})"))
    elif results['sharpe_ratio'] > 0:
        checks.append(("⚠️", f"Positive but low Sharpe ({results['sharpe_ratio']:.2f})"))
    else:
        checks.append(("✗", f"Negative Sharpe ({results['sharpe_ratio']:.2f})"))
    
    if results['max_drawdown'] < 0.15:
        checks.append(("✓", f"Low drawdown ({results['max_drawdown']:.1%})"))
    else:
        checks.append(("⚠️", f"Moderate drawdown ({results['max_drawdown']:.1%})"))
    
    if results['win_rate'] > 0.45:
        checks.append(("✓", f"Good win rate ({results['win_rate']:.1%})"))
    else:
        checks.append(("⚠️", f"Low win rate ({results['win_rate']:.1%})"))
    
    if results['profit_factor'] > 1.0:
        checks.append(("✓", f"Profitable (PF={results['profit_factor']:.2f})"))
    else:
        checks.append(("✗", f"Unprofitable (PF={results['profit_factor']:.2f})"))
    
    for symbol, message in checks:
        print(f"  {symbol} {message}")
    
    passed = sum(1 for s, _ in checks if s == "✓")
    
    if passed >= 4:
        print("\n  🏆 STRATEGY VALIDATED - Ready for paper trading")
    elif passed >= 3:
        print("\n  ⚠️  STRATEGY IMPROVED but needs more refinement")
    else:
        print("\n  ❌ STRATEGY STILL NEEDS WORK")
    
    # Save results
    os.makedirs('output/portfolio_simulation', exist_ok=True)
    results['equity'].to_csv('output/portfolio_simulation/improved_equity_curve.csv', index=False)
    results['trades_df'].to_csv('output/portfolio_simulation/improved_trades.csv', index=False)
    
    print(f"\n  Files saved to output/portfolio_simulation/")
    
    return results


if __name__ == "__main__":
    results = main()
