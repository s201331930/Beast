#!/usr/bin/env python3
"""
CHAMPIONSHIP STRATEGY BACKTESTER
================================
Backtests all technical strategies from 2019 to present on ALL TASI stocks.

Strategies Tested:
1. BB Squeeze Breakout
2. VCP (Volatility Contraction Pattern)
3. RS Momentum (Top Decile)
4. Stage 2 Entry
5. Accumulation + Breakout
6. Cup & Handle Pattern
7. Flat Base Breakout
8. MA Alignment (Golden Cross)
9. Volume Dry-Up + Breakout
10. Combined/Blended Strategy

Output:
- Win rate, profit factor, avg return per strategy
- Best performing strategy identification
- Optimal blended strategy parameters
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf


# =============================================================================
# TASI TICKERS - COMPREHENSIVE LIST
# =============================================================================

TASI_TICKERS = [
    # Banks
    '1180.SR', '1010.SR', '1150.SR', '1140.SR', '1060.SR', '1050.SR',
    '1020.SR', '1030.SR', '1080.SR',
    # Energy
    '2222.SR', '4030.SR', '2381.SR', '2380.SR',
    # Petrochemicals
    '2010.SR', '2290.SR', '2250.SR', '2210.SR', '2001.SR', '2060.SR',
    '2310.SR', '2350.SR', '2330.SR', '2170.SR', '2020.SR', '2190.SR',
    # Materials
    '1211.SR', '1320.SR', '1321.SR', '1302.SR', '1304.SR', '2200.SR',
    '2220.SR', '2240.SR', '2320.SR', '2370.SR',
    # Cement
    '3010.SR', '3020.SR', '3030.SR', '3040.SR', '3050.SR', '3060.SR',
    '3080.SR', '3090.SR',
    # Utilities
    '5110.SR', '2082.SR', '2083.SR',
    # Retail & Consumer
    '4190.SR', '4003.SR', '4240.SR', '4001.SR', '4002.SR', '4004.SR',
    '4007.SR', '4009.SR', '4020.SR', '4031.SR', '4050.SR', '4080.SR',
    '4110.SR', '4140.SR', '4200.SR', '4220.SR', '4250.SR', '4270.SR',
    '4280.SR', '4290.SR',
    # Food
    '2280.SR', '2050.SR', '6002.SR', '6001.SR', '6010.SR', '4071.SR',
    # Telecom
    '7010.SR', '7020.SR', '7030.SR',
    # Real Estate
    '4300.SR', '4310.SR', '4320.SR', '4330.SR', '4331.SR', '4332.SR',
    '4333.SR', '1120.SR',
    # Insurance
    '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR', '8050.SR',
    '8060.SR', '8100.SR', '8120.SR', '8150.SR', '8160.SR', '8180.SR',
    '8200.SR', '8210.SR', '8230.SR', '8240.SR', '8250.SR', '8300.SR',
    '8310.SR',
    # Diversified
    '1111.SR',
]

TASI_TICKERS = list(set(TASI_TICKERS))


# =============================================================================
# TRADE RESULT
# =============================================================================

@dataclass
class Trade:
    ticker: str
    strategy: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    return_pct: float
    holding_days: int
    win: bool
    exit_reason: str  # TARGET/STOP/TIME/SIGNAL


@dataclass 
class StrategyResult:
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_holding_days: float
    best_trade: float
    worst_trade: float


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Calculate all technical indicators for backtesting."""
    
    @staticmethod
    def calculate_bb(close: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_bb_width(close: pd.Series, period: int = 20) -> pd.Series:
        """Bollinger Band width (for squeeze detection)."""
        sma = close.rolling(period).mean()
        std_dev = close.rolling(period).std()
        width = (2 * std_dev) / sma
        return width
    
    @staticmethod
    def calculate_keltner_width(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """Keltner Channel width."""
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        ema = close.ewm(span=period).mean()
        width = (2 * 1.5 * atr) / ema
        return width
    
    @staticmethod
    def detect_squeeze(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Detect BB squeeze (BB inside Keltner)."""
        bb_width = TechnicalIndicators.calculate_bb_width(close)
        kc_width = TechnicalIndicators.calculate_keltner_width(close, high, low)
        return bb_width < kc_width
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 0.001)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        return dx.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()
    
    @staticmethod
    def calculate_accumulation(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line."""
        mfm = ((close - low) - (high - close)) / (high - low + 0.001)
        mfv = mfm * volume
        return mfv.cumsum()
    
    @staticmethod
    def calculate_rs(close: pd.Series, market: pd.Series) -> pd.Series:
        """Relative Strength vs Market."""
        stock_ret = close.pct_change(20)
        market_ret = market.pct_change(20)
        return stock_ret - market_ret


# =============================================================================
# STRATEGY SIGNAL GENERATORS
# =============================================================================

class StrategySignals:
    """Generate entry/exit signals for each strategy."""
    
    @staticmethod
    def bb_squeeze_breakout(df: pd.DataFrame) -> pd.Series:
        """
        BB Squeeze Breakout Strategy:
        - Entry: Price breaks above upper BB after squeeze
        - Squeeze: BB inside Keltner Channel
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        squeeze = TechnicalIndicators.detect_squeeze(close, high, low)
        upper_bb, _, _ = TechnicalIndicators.calculate_bb(close)
        
        # Squeeze for at least 5 days, then breakout
        squeeze_duration = squeeze.rolling(10).sum()
        breakout = (close > upper_bb) & (squeeze_duration.shift(1) >= 5)
        
        return breakout.astype(int)
    
    @staticmethod
    def vcp_pattern(df: pd.DataFrame) -> pd.Series:
        """
        VCP (Volatility Contraction Pattern):
        - Entry: Volatility contracts in series, then price breaks out
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Calculate volatility in 3 periods
        def vol_range(window):
            return (high.rolling(window).max() - low.rolling(window).min()) / close.rolling(window).mean()
        
        vol_1 = vol_range(20)
        vol_2 = vol_range(10).shift(20)
        vol_3 = vol_range(10).shift(10)
        
        # Contracting volatility
        contracting = (vol_1 < vol_2 * 0.8) & (vol_2 < vol_3 * 0.8)
        
        # Near highs
        recent_high = high.rolling(60).max()
        near_high = close > recent_high * 0.95
        
        signal = contracting & near_high
        return signal.astype(int)
    
    @staticmethod
    def rs_momentum(df: pd.DataFrame, market: pd.Series) -> pd.Series:
        """
        RS Momentum Strategy:
        - Entry: Stock in top RS decile + positive momentum
        """
        close = df['close']
        
        # 60-day momentum
        momentum = close.pct_change(60)
        
        # RS vs market
        rs = TechnicalIndicators.calculate_rs(close, market)
        
        # Strong momentum + outperforming market
        signal = (momentum > 0.15) & (rs > 0.05)
        
        return signal.astype(int)
    
    @staticmethod
    def stage2_entry(df: pd.DataFrame) -> pd.Series:
        """
        Stage 2 Entry (Weinstein):
        - Entry: Price crosses above 30-week MA, MA turning up
        """
        close = df['close']
        
        # 30-week MA (150 days)
        ma_150 = close.rolling(150).mean()
        
        # MA slope
        ma_slope = ma_150.diff(20) / ma_150.shift(20)
        
        # Cross above with rising MA
        cross_above = (close > ma_150) & (close.shift(1) <= ma_150.shift(1))
        rising_ma = ma_slope > 0.01
        
        signal = cross_above & rising_ma
        return signal.astype(int)
    
    @staticmethod
    def accumulation_breakout(df: pd.DataFrame) -> pd.Series:
        """
        Accumulation + Breakout:
        - Entry: A/D line rising while price consolidates, then breakout
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # A/D Line
        ad = TechnicalIndicators.calculate_accumulation(close, high, low, volume)
        ad_slope = ad.diff(20)
        
        # Price consolidation (tight range)
        price_range = (high.rolling(20).max() - low.rolling(20).min()) / close
        consolidating = price_range < 0.10
        
        # Breakout
        recent_high = high.rolling(30).max().shift(1)
        breakout = close > recent_high
        
        signal = (ad_slope > 0) & consolidating.shift(1) & breakout
        return signal.astype(int)
    
    @staticmethod
    def cup_handle(df: pd.DataFrame) -> pd.Series:
        """
        Cup & Handle Pattern:
        - U-shaped base followed by small handle, then breakout
        """
        close = df['close']
        high = df['high']
        
        # 60-day lookback for cup
        rolling_high = high.rolling(60).max()
        rolling_low = close.rolling(60).min()
        
        # Cup depth 15-35%
        cup_depth = (rolling_high - rolling_low) / rolling_high
        valid_cup = (cup_depth > 0.15) & (cup_depth < 0.35)
        
        # Near cup high (handle area)
        near_high = close > rolling_high * 0.90
        
        # Handle: small consolidation (5-10 days)
        handle_range = (high.rolling(10).max() - close.rolling(10).min()) / close
        valid_handle = handle_range < 0.08
        
        # Breakout above cup high
        breakout = close > rolling_high.shift(1)
        
        signal = valid_cup & near_high.shift(5) & valid_handle.shift(1) & breakout
        return signal.astype(int)
    
    @staticmethod
    def flat_base_breakout(df: pd.DataFrame) -> pd.Series:
        """
        Flat Base Breakout:
        - Price consolidates in tight range (<15%), then breaks out
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Base: tight range for 20+ days
        base_high = high.rolling(30).max()
        base_low = low.rolling(30).min()
        base_range = (base_high - base_low) / base_low
        
        flat_base = base_range < 0.15
        
        # Breakout above base
        breakout = close > base_high.shift(1)
        
        signal = flat_base.shift(1) & breakout
        return signal.astype(int)
    
    @staticmethod
    def ma_alignment(df: pd.DataFrame) -> pd.Series:
        """
        MA Alignment (Golden Cross):
        - Entry: 20 > 50 > 200, price above all
        """
        close = df['close']
        
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        
        aligned = (ma_20 > ma_50) & (ma_50 > ma_200)
        above_all = (close > ma_20) & (close > ma_50) & (close > ma_200)
        
        # Fresh alignment (just happened)
        fresh = aligned & ~aligned.shift(1)
        
        signal = fresh & above_all
        return signal.astype(int)
    
    @staticmethod
    def volume_dryup_breakout(df: pd.DataFrame) -> pd.Series:
        """
        Volume Dry-Up + Breakout:
        - Entry: Volume contracts during base, then expands on breakout
        """
        close = df['close']
        high = df['high']
        volume = df['volume']
        
        # Volume dry-up
        vol_ma = volume.rolling(50).mean()
        vol_ratio = volume / vol_ma
        dry_up = vol_ratio.rolling(10).mean() < 0.7
        
        # Price near highs
        recent_high = high.rolling(40).max()
        near_high = close > recent_high * 0.95
        
        # Volume expansion on breakout
        vol_expansion = vol_ratio > 1.5
        breakout = close > recent_high.shift(1)
        
        signal = dry_up.shift(1) & near_high.shift(1) & vol_expansion & breakout
        return signal.astype(int)


# =============================================================================
# BACKTESTER
# =============================================================================

class StrategyBacktester:
    """Backtest all strategies on historical data."""
    
    STRATEGIES = [
        'bb_squeeze_breakout',
        'vcp_pattern',
        'rs_momentum',
        'stage2_entry',
        'accumulation_breakout',
        'cup_handle',
        'flat_base_breakout',
        'ma_alignment',
        'volume_dryup_breakout',
    ]
    
    def __init__(self, 
                 start_date: str = '2019-01-01',
                 end_date: str = None,
                 holding_days: int = 20,
                 stop_loss_pct: float = 0.08,
                 take_profit_pct: float = 0.15):
        
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.holding_days = holding_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.market_data: Optional[pd.DataFrame] = None
        self.all_trades: Dict[str, List[Trade]] = {s: [] for s in self.STRATEGIES}
        self.results: Dict[str, StrategyResult] = {}
    
    def fetch_data(self) -> None:
        """Fetch all historical data."""
        print(f"Fetching data from {self.start_date} to {self.end_date}...")
        
        # Fetch market index (TASI)
        try:
            tasi = yf.Ticker('^TASI.SR')
            self.market_data = tasi.history(start=self.start_date, end=self.end_date)
            if len(self.market_data) < 100:
                # Use Saudi Aramco as proxy
                aramco = yf.Ticker('2222.SR')
                self.market_data = aramco.history(start=self.start_date, end=self.end_date)
            self.market_data.columns = [c.lower() for c in self.market_data.columns]
        except:
            pass
        
        # Fetch all stocks
        loaded = 0
        for i, ticker in enumerate(TASI_TICKERS):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                if len(df) >= 250:  # At least 1 year of data
                    df.columns = [c.lower() for c in df.columns]
                    df.index = df.index.tz_localize(None)
                    self.stock_data[ticker] = df
                    loaded += 1
            except:
                pass
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(TASI_TICKERS)} | Loaded: {loaded}")
        
        print(f"Loaded {loaded} stocks with sufficient history")
        
        # Align market data
        if self.market_data is not None and len(self.market_data) > 0:
            self.market_data.index = self.market_data.index.tz_localize(None)
    
    def simulate_trade(self, df: pd.DataFrame, entry_idx: int, ticker: str, strategy: str) -> Optional[Trade]:
        """Simulate a single trade from entry point."""
        if entry_idx >= len(df) - 1:
            return None
        
        entry_date = df.index[entry_idx].strftime('%Y-%m-%d')
        entry_price = df['close'].iloc[entry_idx]
        
        # Simulate forward
        for i in range(entry_idx + 1, min(entry_idx + self.holding_days + 1, len(df))):
            current_price = df['close'].iloc[i]
            high_price = df['high'].iloc[i]
            low_price = df['low'].iloc[i]
            
            # Check stop loss
            if low_price <= entry_price * (1 - self.stop_loss_pct):
                exit_price = entry_price * (1 - self.stop_loss_pct)
                return Trade(
                    ticker=ticker,
                    strategy=strategy,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=df.index[i].strftime('%Y-%m-%d'),
                    exit_price=exit_price,
                    return_pct=(exit_price / entry_price - 1) * 100,
                    holding_days=i - entry_idx,
                    win=False,
                    exit_reason='STOP'
                )
            
            # Check take profit
            if high_price >= entry_price * (1 + self.take_profit_pct):
                exit_price = entry_price * (1 + self.take_profit_pct)
                return Trade(
                    ticker=ticker,
                    strategy=strategy,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=df.index[i].strftime('%Y-%m-%d'),
                    exit_price=exit_price,
                    return_pct=(exit_price / entry_price - 1) * 100,
                    holding_days=i - entry_idx,
                    win=True,
                    exit_reason='TARGET'
                )
        
        # Time exit
        exit_idx = min(entry_idx + self.holding_days, len(df) - 1)
        exit_price = df['close'].iloc[exit_idx]
        return_pct = (exit_price / entry_price - 1) * 100
        
        return Trade(
            ticker=ticker,
            strategy=strategy,
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=df.index[exit_idx].strftime('%Y-%m-%d'),
            exit_price=exit_price,
            return_pct=return_pct,
            holding_days=exit_idx - entry_idx,
            win=return_pct > 0,
            exit_reason='TIME'
        )
    
    def backtest_strategy(self, strategy_name: str) -> None:
        """Backtest a single strategy across all stocks."""
        print(f"  Backtesting {strategy_name}...")
        
        signal_func = getattr(StrategySignals, strategy_name)
        trades = []
        
        for ticker, df in self.stock_data.items():
            try:
                # Generate signals
                if strategy_name == 'rs_momentum' and self.market_data is not None:
                    market_close = self.market_data['close'].reindex(df.index, method='ffill')
                    signals = signal_func(df, market_close)
                else:
                    signals = signal_func(df)
                
                # Find entry points
                entry_points = signals[signals == 1].index
                
                last_exit_date = None
                for entry_date in entry_points:
                    # Skip if overlapping with previous trade
                    if last_exit_date and entry_date <= last_exit_date:
                        continue
                    
                    entry_idx = df.index.get_loc(entry_date)
                    trade = self.simulate_trade(df, entry_idx, ticker, strategy_name)
                    
                    if trade:
                        trades.append(trade)
                        last_exit_date = pd.Timestamp(trade.exit_date) + timedelta(days=5)
                        
            except Exception as e:
                continue
        
        self.all_trades[strategy_name] = trades
    
    def calculate_results(self, strategy_name: str) -> StrategyResult:
        """Calculate performance metrics for a strategy."""
        trades = self.all_trades[strategy_name]
        
        if not trades:
            return StrategyResult(
                name=strategy_name,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_return=0, avg_winner=0, avg_loser=0,
                profit_factor=0, total_return=0, max_drawdown=0,
                sharpe_ratio=0, avg_holding_days=0, best_trade=0, worst_trade=0
            )
        
        returns = [t.return_pct for t in trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        
        win_rate = len(winners) / len(returns) * 100 if returns else 0
        avg_return = np.mean(returns) if returns else 0
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_return = sum(returns)
        
        # Drawdown calculation
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / self.holding_days)
        else:
            sharpe = 0
        
        avg_holding = np.mean([t.holding_days for t in trades])
        
        return StrategyResult(
            name=strategy_name,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 1),
            avg_return=round(avg_return, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            profit_factor=round(profit_factor, 2),
            total_return=round(total_return, 1),
            max_drawdown=round(max_drawdown, 1),
            sharpe_ratio=round(sharpe, 2),
            avg_holding_days=round(avg_holding, 1),
            best_trade=round(max(returns), 2) if returns else 0,
            worst_trade=round(min(returns), 2) if returns else 0
        )
    
    def run_backtest(self) -> None:
        """Run complete backtest."""
        self.fetch_data()
        
        print("\nBacktesting all strategies...")
        for strategy in self.STRATEGIES:
            self.backtest_strategy(strategy)
            self.results[strategy] = self.calculate_results(strategy)
    
    def generate_report(self) -> str:
        """Generate comprehensive backtest report."""
        lines = []
        
        lines.append("=" * 120)
        lines.append("🏆 CHAMPIONSHIP STRATEGY BACKTEST RESULTS")
        lines.append(f"   Period: {self.start_date} to {self.end_date}")
        lines.append(f"   Holding Period: {self.holding_days} days | Stop Loss: {self.stop_loss_pct*100:.0f}% | Take Profit: {self.take_profit_pct*100:.0f}%")
        lines.append(f"   Stocks Tested: {len(self.stock_data)}")
        lines.append("=" * 120)
        
        # Strategy Comparison Table
        lines.append(f"""
{'='*120}
📊 STRATEGY COMPARISON
{'='*120}

  {'Strategy':<25} {'Trades':>7} {'Win%':>6} {'AvgRet':>8} {'PF':>6} {'Total%':>8} {'Sharpe':>7} {'MaxDD':>7} {'Best':>7} {'Worst':>7}
  {'-'*115}""")
        
        # Sort by profit factor
        sorted_results = sorted(self.results.values(), key=lambda x: x.profit_factor, reverse=True)
        
        for r in sorted_results:
            emoji = "🥇" if r == sorted_results[0] else "🥈" if r == sorted_results[1] else "🥉" if r == sorted_results[2] else "  "
            lines.append(f"  {emoji}{r.name:<23} {r.total_trades:>7} {r.win_rate:>5.1f}% {r.avg_return:>+7.2f}% "
                        f"{r.profit_factor:>5.2f} {r.total_return:>+7.1f}% {r.sharpe_ratio:>6.2f} {r.max_drawdown:>6.1f}% "
                        f"{r.best_trade:>+6.1f}% {r.worst_trade:>+6.1f}%")
        
        # Top 3 Strategies Analysis
        lines.append(f"""

{'='*120}
🥇 TOP 3 STRATEGIES - DETAILED ANALYSIS
{'='*120}
""")
        
        for i, r in enumerate(sorted_results[:3]):
            medal = ["🥇 GOLD", "🥈 SILVER", "🥉 BRONZE"][i]
            trades = self.all_trades[r.name]
            
            # Win by exit reason
            target_exits = len([t for t in trades if t.exit_reason == 'TARGET'])
            stop_exits = len([t for t in trades if t.exit_reason == 'STOP'])
            time_exits = len([t for t in trades if t.exit_reason == 'TIME'])
            
            lines.append(f"""
  {medal}: {r.name.upper().replace('_', ' ')}
  {'─'*100}
  
  PERFORMANCE:
    Total Trades:       {r.total_trades}
    Win Rate:           {r.win_rate:.1f}%
    Profit Factor:      {r.profit_factor:.2f}
    Avg Return/Trade:   {r.avg_return:+.2f}%
    Total Return:       {r.total_return:+.1f}%
    Sharpe Ratio:       {r.sharpe_ratio:.2f}
  
  WIN/LOSS ANALYSIS:
    Avg Winner:         {r.avg_winner:+.2f}%
    Avg Loser:          {r.avg_loser:+.2f}%
    Best Trade:         {r.best_trade:+.2f}%
    Worst Trade:        {r.worst_trade:+.2f}%
    Max Drawdown:       {r.max_drawdown:.1f}%
  
  EXIT ANALYSIS:
    Target Exits:       {target_exits} ({target_exits/max(1,r.total_trades)*100:.1f}%)
    Stop Exits:         {stop_exits} ({stop_exits/max(1,r.total_trades)*100:.1f}%)
    Time Exits:         {time_exits} ({time_exits/max(1,r.total_trades)*100:.1f}%)
""")
        
        # Strategies to DROP
        poor_strategies = [r for r in sorted_results if r.profit_factor < 1.0 or r.win_rate < 40]
        
        if poor_strategies:
            lines.append(f"""
{'='*120}
❌ STRATEGIES TO DROP (PF < 1.0 or Win Rate < 40%)
{'='*120}
""")
            for r in poor_strategies:
                lines.append(f"  • {r.name}: PF={r.profit_factor:.2f}, Win={r.win_rate:.1f}%, Avg={r.avg_return:+.2f}%")
        
        # Blended Strategy Recommendation
        good_strategies = [r for r in sorted_results if r.profit_factor >= 1.2 and r.win_rate >= 45]
        
        lines.append(f"""
{'='*120}
✅ RECOMMENDED BLENDED STRATEGY
{'='*120}

  Based on backtest results, the optimal blended strategy combines:
""")
        
        if good_strategies:
            for r in good_strategies:
                weight = min(40, r.profit_factor * 15)
                lines.append(f"    • {r.name}: {weight:.0f}% weight (PF={r.profit_factor:.2f}, Win={r.win_rate:.1f}%)")
        else:
            lines.append("    No strategies met the quality threshold (PF >= 1.2, Win >= 45%)")
        
        # Final Recommendation
        best = sorted_results[0] if sorted_results else None
        
        lines.append(f"""

{'='*120}
🎯 FINAL RECOMMENDATION
{'='*120}

  BEST SINGLE STRATEGY: {best.name if best else 'None'}
    - Profit Factor: {best.profit_factor if best else 0:.2f}
    - Win Rate: {best.win_rate if best else 0:.1f}%
    - Avg Return: {best.avg_return if best else 0:+.2f}% per trade

  IMPLEMENTATION:
    - Use {best.name if best else 'N/A'} as primary signal
    - Apply market regime filter (BULL market only for full exposure)
    - Position size: {min(5, 100/max(1, abs(best.worst_trade) if best else 10)):.1f}% per trade (based on max loss)
    - Stop loss: {self.stop_loss_pct*100:.0f}%
    - Take profit: {self.take_profit_pct*100:.0f}%

{'='*120}
END OF BACKTEST REPORT
{'='*120}
""")
        
        return "\n".join(lines)
    
    def save_results(self, output_dir: str = "output/backtest") -> str:
        """Save backtest results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save report
        report = self.generate_report()
        report_path = f"{output_dir}/strategy_backtest_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save trades to CSV
        all_trades = []
        for strategy, trades in self.all_trades.items():
            for t in trades:
                all_trades.append(asdict(t))
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(f"{output_dir}/all_trades_{timestamp}.csv", index=False)
        
        # Save summary
        summary = {
            'backtest_date': timestamp,
            'period': f"{self.start_date} to {self.end_date}",
            'stocks_tested': len(self.stock_data),
            'strategies': {name: asdict(result) for name, result in self.results.items()}
        }
        
        with open(f"{output_dir}/backtest_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("🏆 CHAMPIONSHIP STRATEGY BACKTESTER")
    print("   Testing all strategies from 2019 to present")
    print("=" * 70 + "\n")
    
    backtester = StrategyBacktester(
        start_date='2019-01-01',
        holding_days=20,
        stop_loss_pct=0.08,
        take_profit_pct=0.15
    )
    
    backtester.run_backtest()
    report = backtester.generate_report()
    print(report)
    
    filepath = backtester.save_results()
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
