#!/usr/bin/env python3
"""
FULL MARKET SCANNER
===================
Scans ALL TASI (Saudi Stock Exchange) stocks and selects the TOP 10
based on scientific screening criteria.

Fetches actual company names from Yahoo Finance to ensure accuracy.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# COMPLETE TASI TICKER LIST
# =============================================================================

# All known TASI tickers - names will be fetched from Yahoo Finance
TASI_TICKERS = [
    # Banks
    '1180.SR', '1010.SR', '1150.SR', '1140.SR', '1060.SR', '1050.SR',
    '1020.SR', '1030.SR', '1080.SR', '1182.SR',
    
    # Energy & Petrochemicals
    '2222.SR', '2030.SR', '4030.SR', '2381.SR', '2380.SR',
    '2010.SR', '2290.SR', '2250.SR', '2210.SR', '2001.SR',
    '2060.SR', '2310.SR', '2350.SR', '2330.SR', '2170.SR',
    '2020.SR', '2190.SR',
    
    # Materials & Industrials
    '1211.SR', '1320.SR', '1321.SR', '1302.SR', '1304.SR',
    '2200.SR', '2220.SR', '2240.SR', '2320.SR', '2370.SR',
    
    # Cement
    '3010.SR', '3020.SR', '3030.SR', '3040.SR', '3050.SR',
    '3060.SR', '3080.SR', '3090.SR', '3091.SR', '3001.SR',
    '3002.SR', '3003.SR', '3004.SR',
    
    # Utilities
    '5110.SR', '2082.SR', '2083.SR',
    
    # Retail & Consumer
    '4190.SR', '4003.SR', '4240.SR', '4001.SR', '4002.SR',
    '4004.SR', '4005.SR', '4006.SR', '4007.SR', '4008.SR',
    '4009.SR', '4020.SR', '4031.SR', '4040.SR', '4050.SR',
    '4051.SR', '4061.SR', '4080.SR', '4110.SR', '4130.SR',
    '4140.SR', '4160.SR', '4180.SR', '4200.SR', '4210.SR',
    '4220.SR', '4230.SR', '4250.SR', '4260.SR', '4261.SR',
    '4270.SR', '4280.SR', '4290.SR', '4291.SR', '4292.SR',
    
    # Food & Agriculture
    '2280.SR', '2050.SR', '6002.SR', '6001.SR', '6010.SR',
    '6020.SR', '6040.SR', '6050.SR', '6060.SR', '6070.SR',
    '4071.SR',
    
    # Telecom
    '7010.SR', '7020.SR', '7030.SR', '7040.SR',
    
    # Real Estate
    '4300.SR', '4310.SR', '4320.SR', '4321.SR', '4322.SR',
    '4323.SR', '4324.SR', '4330.SR', '4331.SR', '4332.SR',
    '4333.SR', '4334.SR', '4336.SR', '4337.SR', '4338.SR',
    '4339.SR', '4340.SR', '4342.SR', '4344.SR', '4347.SR',
    '1120.SR',
    
    # Insurance
    '8010.SR', '8012.SR', '8020.SR', '8030.SR', '8040.SR',
    '8050.SR', '8060.SR', '8070.SR', '8080.SR', '8100.SR',
    '8120.SR', '8150.SR', '8160.SR', '8170.SR', '8180.SR',
    '8190.SR', '8200.SR', '8210.SR', '8230.SR', '8240.SR',
    '8250.SR', '8260.SR', '8270.SR', '8280.SR', '8300.SR',
    '8310.SR', '8311.SR',
    
    # Diversified
    '1111.SR', '1183.SR', '4081.SR', '4082.SR',
]

# Remove duplicates
TASI_TICKERS = list(set(TASI_TICKERS))


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'regime_ma_period': 50,
    'leverage_bull': 1.5,
    'leverage_bear': 0.5,
    'max_stocks': 10,
    'min_score': 50,
    'max_position_pct': 0.15,
    'min_position_pct': 0.05,
}


# =============================================================================
# SCIENTIFIC SCREENER
# =============================================================================

@dataclass
class StockScore:
    ticker: str
    name: str  # Fetched from Yahoo Finance
    overall_score: float
    recommendation: str
    momentum_score: float
    trend_score: float
    beta_score: float
    volatility_score: float
    liquidity_score: float
    retail_score: float
    regime_score: float
    hurst_exponent: float
    beta: float
    adx: float
    volatility: float
    momentum_20d: float
    is_trending: bool
    is_high_beta: bool
    in_bull_regime: bool


class ScientificScreener:
    """Scientific screening system."""
    
    OPTIMAL_HURST_MIN = 0.55
    OPTIMAL_BETA_MIN = 1.2
    OPTIMAL_ADX_MIN = 20
    OPTIMAL_VOLATILITY_MIN = 0.30
    OPTIMAL_VOLATILITY_MAX = 1.50
    
    WEIGHTS = {
        'momentum': 0.20,
        'trend': 0.20,
        'beta': 0.20,
        'volatility': 0.15,
        'liquidity': 0.10,
        'retail': 0.10,
        'regime': 0.05
    }
    
    def __init__(self, df: pd.DataFrame, ticker: str, name: str):
        self.df = df.copy()
        self.ticker = ticker
        self.name = name
    
    def calculate_hurst(self, series: pd.Series, min_w: int = 10, max_w: int = 100) -> float:
        series = series.dropna().values
        n = len(series)
        if n < max_w:
            max_w = max(min_w + 5, n // 2)
        
        rs_list, n_list = [], []
        for window in range(min_w, max_w + 1, 5):
            rs_values = []
            for start in range(0, n - window, window // 2):
                subset = series[start:start + window]
                mean_adj = subset - np.mean(subset)
                cumsum = np.cumsum(mean_adj)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(subset, ddof=1)
                if S > 0:
                    rs_values.append(R / S)
            if rs_values:
                rs_list.append(np.mean(rs_values))
                n_list.append(window)
        
        if len(rs_list) > 2:
            log_n = np.log(n_list)
            log_rs = np.log(rs_list)
            slope, _ = np.polyfit(log_n, log_rs, 1)
            return slope
        return 0.5
    
    def calculate_adx(self, period: int = 14) -> float:
        try:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 0.001)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return adx.iloc[-1] if not adx.empty and not np.isnan(adx.iloc[-1]) else 25
        except:
            return 25
    
    def calculate_momentum_score(self) -> Tuple[float, float]:
        close = self.df['close']
        momentum_20d = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
        momentum_60d = close.pct_change(60).iloc[-1] if len(close) > 60 else 0
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
        
        price = close.iloc[-1]
        
        factors = [
            momentum_20d > 0,
            momentum_60d > 0,
            price > sma_20,
            price > sma_50,
            price > sma_200,
            sma_20 > sma_50,
        ]
        
        score = sum(factors) / len(factors) * 100
        return score, momentum_20d
    
    def calculate_trend_score(self) -> Tuple[float, float, float, bool]:
        close = self.df['close']
        hurst = self.calculate_hurst(close)
        adx = self.calculate_adx()
        
        hurst_score = min(100, max(0, (hurst - 0.3) / 0.4 * 100))
        adx_score = min(100, max(0, adx / 50 * 100))
        
        trend_score = hurst_score * 0.5 + adx_score * 0.5
        is_trending = hurst > self.OPTIMAL_HURST_MIN and adx > self.OPTIMAL_ADX_MIN
        
        return trend_score, hurst, adx, is_trending
    
    def calculate_beta_score(self) -> Tuple[float, float, bool]:
        returns = self.df['close'].pct_change().dropna()
        stock_vol = returns.std()
        beta = min(4.0, stock_vol / 0.01)
        
        if beta < 0.5:
            score = beta / 0.5 * 30
        elif beta < self.OPTIMAL_BETA_MIN:
            score = 30 + (beta - 0.5) / 0.7 * 30
        elif beta <= 3.0:
            score = 60 + (beta - 1.2) / 1.8 * 40
        else:
            score = 100 - (beta - 3.0) * 10
        
        score = min(100, max(0, score))
        is_high_beta = beta >= self.OPTIMAL_BETA_MIN
        return score, beta, is_high_beta
    
    def calculate_volatility_score(self) -> Tuple[float, float]:
        returns = self.df['close'].pct_change().dropna()
        volatility = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else returns.std() * np.sqrt(252)
        
        if volatility < self.OPTIMAL_VOLATILITY_MIN:
            score = volatility / self.OPTIMAL_VOLATILITY_MIN * 50
        elif volatility > self.OPTIMAL_VOLATILITY_MAX:
            score = max(0, 100 - (volatility - self.OPTIMAL_VOLATILITY_MAX) * 100)
        else:
            score = 70 + (volatility - 0.30) / 0.70 * 30
        
        return min(100, max(0, score)), volatility
    
    def calculate_liquidity_score(self) -> float:
        volume = self.df['volume']
        close = self.df['close']
        dollar_volume = (close * volume).tail(20).mean()
        
        if dollar_volume >= 100_000_000:
            return 100
        elif dollar_volume >= 50_000_000:
            return 90
        elif dollar_volume >= 10_000_000:
            return 70
        elif dollar_volume >= 1_000_000:
            return 50
        else:
            return max(10, dollar_volume / 1_000_000 * 50)
    
    def calculate_retail_score(self) -> float:
        close = self.df['close']
        volume = self.df['volume']
        price = close.iloc[-1]
        
        if price < 20:
            price_score = 100
        elif price < 50:
            price_score = 85
        elif price < 100:
            price_score = 70
        else:
            price_score = 50
        
        try:
            volume_zscore = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 0.001)
            spike_freq = (volume_zscore > 2).tail(60).sum() / 60 * 100
        except:
            spike_freq = 50
        
        momentum = close.pct_change(20).iloc[-1] if len(close) > 20 else 0
        momentum_score = min(100, max(0, 50 + momentum * 200))
        
        return price_score * 0.4 + spike_freq * 0.3 + momentum_score * 0.3
    
    def calculate_regime_score(self) -> Tuple[float, bool]:
        close = self.df['close']
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
        price = close.iloc[-1]
        
        golden_cross = sma_50 > sma_200
        above_200 = price > sma_200
        
        score = 0
        if golden_cross:
            score += 50
        if above_200:
            score += 50
        
        return score, golden_cross and above_200
    
    def analyze(self) -> StockScore:
        momentum_score, momentum_20d = self.calculate_momentum_score()
        trend_score, hurst, adx, is_trending = self.calculate_trend_score()
        beta_score, beta, is_high_beta = self.calculate_beta_score()
        vol_score, volatility = self.calculate_volatility_score()
        liq_score = self.calculate_liquidity_score()
        retail_score = self.calculate_retail_score()
        regime_score, in_bull = self.calculate_regime_score()
        
        overall = (
            self.WEIGHTS['momentum'] * momentum_score +
            self.WEIGHTS['trend'] * trend_score +
            self.WEIGHTS['beta'] * beta_score +
            self.WEIGHTS['volatility'] * vol_score +
            self.WEIGHTS['liquidity'] * liq_score +
            self.WEIGHTS['retail'] * retail_score +
            self.WEIGHTS['regime'] * regime_score
        )
        
        if overall >= 75:
            rec = 'EXCELLENT'
        elif overall >= 60:
            rec = 'GOOD'
        elif overall >= 45:
            rec = 'MODERATE'
        else:
            rec = 'POOR'
        
        return StockScore(
            ticker=self.ticker,
            name=self.name,
            overall_score=overall,
            recommendation=rec,
            momentum_score=momentum_score,
            trend_score=trend_score,
            beta_score=beta_score,
            volatility_score=vol_score,
            liquidity_score=liq_score,
            retail_score=retail_score,
            regime_score=regime_score,
            hurst_exponent=hurst,
            beta=beta,
            adx=adx,
            volatility=volatility,
            momentum_20d=momentum_20d,
            is_trending=is_trending,
            is_high_beta=is_high_beta,
            in_bull_regime=in_bull
        )


# =============================================================================
# POSITION DATA
# =============================================================================

@dataclass
class Position:
    ticker: str
    name: str
    suitability_score: float
    recommendation: str
    current_price: float
    ma_50: float
    price_vs_ma_pct: float
    daily_change_pct: float
    target_weight_pct: float
    target_value: float
    target_shares: int
    regime_change_price: float
    stock_regime: str
    hurst: float
    beta: float
    adx: float
    volatility: float
    momentum_20d: float


@dataclass
class RegimeRecord:
    start_date: str
    end_date: Optional[str]
    regime: str
    leverage: float
    days: int
    status: str


# =============================================================================
# FULL MARKET SCANNER
# =============================================================================

class FullMarketScanner:
    """Scans entire TASI market and selects top stocks."""
    
    def __init__(self, capital: float = 1_000_000):
        self.capital = capital
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.stock_names: Dict[str, str] = {}  # Ticker -> Actual company name
        self.stock_scores: Dict[str, StockScore] = {}
        self.all_scores: List[StockScore] = []
        self.selected_stocks: List[str] = []
        self.positions: List[Position] = []
        self.tracker_file = "output/production/performance_tracker.json"
        self.regime_history: List[RegimeRecord] = []
        self.scan_time = None
        self.current_regime = None
        self.current_leverage = None
        self.stocks_screened = 0
        self.stocks_loaded = 0
    
    def fetch_all_data(self) -> None:
        """Fetch data for ALL TASI stocks and get actual company names."""
        total = len(TASI_TICKERS)
        print(f"Fetching data for {total} TASI tickers...")
        print("This may take a few minutes...\n")
        
        loaded = 0
        failed = 0
        
        for i, ticker in enumerate(TASI_TICKERS):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")
                
                if len(df) >= 60:
                    df.columns = [c.lower() for c in df.columns]
                    self.stock_data[ticker] = df
                    
                    # Get actual company name from Yahoo Finance
                    try:
                        info = stock.info
                        # Try different fields for company name
                        name = info.get('longName') or info.get('shortName') or info.get('displayName') or ticker
                        # Clean up the name
                        name = name.replace('Company', 'Co').replace('Corporation', 'Corp')
                        if len(name) > 35:
                            name = name[:32] + '...'
                        self.stock_names[ticker] = name
                    except:
                        self.stock_names[ticker] = ticker
                    
                    loaded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
            
            # Progress indicator
            if (i + 1) % 20 == 0 or i == total - 1:
                print(f"  Progress: {i+1}/{total} | Loaded: {loaded} | Failed: {failed}")
        
        self.stocks_loaded = loaded
        print(f"\nTotal stocks with valid data: {loaded}")
    
    def screen_all_stocks(self) -> None:
        """Run scientific screening on all stocks."""
        print("\n" + "=" * 70)
        print("SCIENTIFIC SCREENING - ALL TASI STOCKS")
        print("=" * 70)
        
        self.all_scores = []
        
        for ticker, df in self.stock_data.items():
            name = self.stock_names.get(ticker, ticker)
            try:
                screener = ScientificScreener(df, ticker, name)
                score = screener.analyze()
                self.stock_scores[ticker] = score
                self.all_scores.append(score)
            except Exception as e:
                pass
        
        self.stocks_screened = len(self.all_scores)
        
        # Sort by score
        self.all_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Select top N with minimum score
        qualified = [s for s in self.all_scores if s.overall_score >= CONFIG['min_score']]
        self.selected_stocks = [s.ticker for s in qualified[:CONFIG['max_stocks']]]
        
        print(f"\nScreening Results:")
        print(f"  Total stocks screened: {self.stocks_screened}")
        print(f"  Score >= {CONFIG['min_score']}: {len(qualified)}")
        print(f"  Selected (top {CONFIG['max_stocks']}): {len(self.selected_stocks)}")
    
    def detect_regime(self) -> Tuple[str, float]:
        if not self.selected_stocks:
            return "UNKNOWN", 1.0
        
        proxy = self.selected_stocks[0]
        df = self.stock_data[proxy]
        close = df['close']
        ma_50 = close.rolling(CONFIG['regime_ma_period']).mean()
        
        price = close.iloc[-1]
        ma = ma_50.iloc[-1]
        
        regime = "BULL" if price > ma else "BEAR"
        leverage = CONFIG['leverage_bull'] if regime == "BULL" else CONFIG['leverage_bear']
        
        return regime, leverage
    
    def calculate_positions(self) -> None:
        if not self.selected_stocks:
            return
        
        base_weight = 1.0 / len(self.selected_stocks)
        leveraged_weight = base_weight * self.current_leverage
        
        for ticker in self.selected_stocks:
            df = self.stock_data[ticker]
            score = self.stock_scores[ticker]
            
            close = df['close']
            ma_50 = close.rolling(50).mean()
            
            price = close.iloc[-1]
            ma = ma_50.iloc[-1]
            prev_close = close.iloc[-2] if len(close) > 1 else price
            
            stock_regime = "BULL" if price > ma else "BEAR"
            
            weight_factor = score.overall_score / 70
            adjusted_weight = leveraged_weight * weight_factor
            adjusted_weight = max(CONFIG['min_position_pct'], 
                                 min(CONFIG['max_position_pct'], adjusted_weight))
            
            target_value = self.capital * adjusted_weight
            target_shares = int(target_value / price)
            
            self.positions.append(Position(
                ticker=ticker,
                name=score.name,
                suitability_score=score.overall_score,
                recommendation=score.recommendation,
                current_price=round(price, 2),
                ma_50=round(ma, 2),
                price_vs_ma_pct=round((price / ma - 1) * 100, 2),
                daily_change_pct=round((price / prev_close - 1) * 100, 2),
                target_weight_pct=round(adjusted_weight * 100, 2),
                target_value=round(target_value, 2),
                target_shares=target_shares,
                regime_change_price=round(ma, 2),
                stock_regime=stock_regime,
                hurst=score.hurst_exponent,
                beta=score.beta,
                adx=score.adx,
                volatility=score.volatility,
                momentum_20d=score.momentum_20d
            ))
        
        self.positions.sort(key=lambda x: x.suitability_score, reverse=True)
    
    def load_tracker(self) -> None:
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                self.regime_history = [RegimeRecord(**r) for r in data.get('regime_history', [])]
            except:
                pass
    
    def save_tracker(self) -> None:
        os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
        data = {
            'last_updated': datetime.now().isoformat(),
            'current_regime': self.current_regime,
            'current_leverage': self.current_leverage,
            'regime_history': [asdict(r) for r in self.regime_history],
            'selected_stocks': self.selected_stocks,
            'stocks_screened': self.stocks_screened
        }
        with open(self.tracker_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_tracker(self) -> None:
        today = datetime.now().strftime('%Y-%m-%d')
        
        if not self.regime_history:
            self.regime_history.append(RegimeRecord(
                start_date=today,
                end_date=None,
                regime=self.current_regime,
                leverage=self.current_leverage,
                days=1,
                status='ACTIVE'
            ))
        else:
            current = self.regime_history[-1]
            if current.regime != self.current_regime:
                current.end_date = today
                current.status = 'CLOSED'
                self.regime_history.append(RegimeRecord(
                    start_date=today,
                    end_date=None,
                    regime=self.current_regime,
                    leverage=self.current_leverage,
                    days=1,
                    status='ACTIVE'
                ))
            else:
                current.days += 1
        
        self.save_tracker()
    
    def run(self) -> str:
        """Run full market scan."""
        self.scan_time = datetime.now()
        self.load_tracker()
        
        # Fetch ALL data
        self.fetch_all_data()
        
        # Screen ALL stocks
        self.screen_all_stocks()
        
        # Detect regime
        self.current_regime, self.current_leverage = self.detect_regime()
        
        # Calculate positions
        self.calculate_positions()
        
        # Update tracker
        self.update_tracker()
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> str:
        """Generate comprehensive report."""
        lines = []
        
        lines.append("=" * 110)
        lines.append("🇸🇦 FULL TASI MARKET SCAN - SCIENTIFIC SCREENING REPORT")
        lines.append(f"   Generated: {self.scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("   Company names fetched directly from Yahoo Finance")
        lines.append("=" * 110)
        
        lines.append(f"""
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FULL MARKET SCREENING SUMMARY                                                                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Total TASI Tickers Scanned:     {len(TASI_TICKERS):>5}                                                                       │
│  Stocks with Valid Data:         {self.stocks_loaded:>5}                                                                       │
│  Stocks Successfully Screened:   {self.stocks_screened:>5}                                                                       │
│  Stocks Passing Threshold (≥{CONFIG['min_score']}): {len([s for s in self.all_scores if s.overall_score >= CONFIG['min_score']]):>5}                                                                       │
│  Final Selection (Top 10):       {len(self.selected_stocks):>5}                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
""")
        
        # Top 20 Stocks by Score
        lines.append("=" * 110)
        lines.append("📊 TOP 20 STOCKS BY SCIENTIFIC SCORE (Full Market Ranking)")
        lines.append("=" * 110)
        
        lines.append(f"\n  {'Rank':<5} {'Ticker':<10} {'Company Name':<35} {'Score':>7} {'Rec':<10} {'Hurst':>6} {'Beta':>5} {'Vol':>6} {'Mom':>7}")
        lines.append("  " + "-" * 105)
        
        for i, score in enumerate(self.all_scores[:20]):
            selected = "✓" if score.ticker in self.selected_stocks else " "
            name_display = score.name[:35] if len(score.name) <= 35 else score.name[:32] + '...'
            lines.append(f"  {i+1:>3}{selected} {score.ticker:<10} {name_display:<35} {score.overall_score:>6.1f} "
                        f"{score.recommendation:<10} {score.hurst_exponent:>6.2f} {score.beta:>5.2f} "
                        f"{score.volatility:>5.0%} {score.momentum_20d:>+6.1%}")
        
        # Current Regime
        regime_emoji = "🟢" if self.current_regime == "BULL" else "🔴"
        proxy = self.selected_stocks[0] if self.selected_stocks else "N/A"
        proxy_name = self.stock_names.get(proxy, proxy)
        proxy_price = self.stock_data[proxy]['close'].iloc[-1] if proxy in self.stock_data else 0
        proxy_ma = self.stock_data[proxy]['close'].rolling(50).mean().iloc[-1] if proxy in self.stock_data else 0
        
        lines.append(f"""

{'='*110}
{regime_emoji} CURRENT MARKET REGIME: {self.current_regime}
{'='*110}

  Market Proxy: {proxy} ({proxy_name})
    Current Price:        {proxy_price:.2f} SAR
    50-Day MA:            {proxy_ma:.2f} SAR
    Price vs MA:          {(proxy_price/proxy_ma-1)*100:+.2f}%
  
  LEVERAGE SETTING:
    Current Regime:       {self.current_regime}
    Leverage:             {self.current_leverage}x
    Total Exposure:       {self.capital * self.current_leverage:,.0f} SAR ({self.current_leverage*100:.0f}% of capital)
  
  REGIME CHANGE TRIGGER:
    If price crosses {proxy_ma:.2f} SAR → Switch leverage
""")
        
        # Selected Top 10 Positions
        lines.append("=" * 110)
        lines.append(f"💰 SELECTED TOP {len(self.positions)} POSITIONS (From {self.stocks_screened} Screened)")
        lines.append("=" * 110)
        
        lines.append(f"\n  Capital: {self.capital:,.0f} SAR | Leverage: {self.current_leverage}x | Total Exposure: {self.capital * self.current_leverage:,.0f} SAR\n")
        
        lines.append(f"  {'Ticker':<10} {'Company Name':<30} {'Score':>6} {'Price':>10} {'Weight':>8} {'Shares':>8} {'Value':>12} {'Regime':<6}")
        lines.append("  " + "-" * 105)
        
        total_value = 0
        for p in self.positions:
            regime_icon = "🟢" if p.stock_regime == "BULL" else "🔴"
            name_display = p.name[:30] if len(p.name) <= 30 else p.name[:27] + '...'
            lines.append(f"  {p.ticker:<10} {name_display:<30} {p.suitability_score:>5.1f} {p.current_price:>10.2f} "
                        f"{p.target_weight_pct:>7.1f}% {p.target_shares:>8} {p.target_value:>12,.0f} {regime_icon}")
            total_value += p.target_value
        
        lines.append("  " + "-" * 105)
        total_weight = sum(p.target_weight_pct for p in self.positions)
        lines.append(f"  {'TOTAL':<10} {'':<30} {'':<6} {'':<10} {total_weight:>7.1f}% {'':<8} {total_value:>12,.0f}")
        
        # Detailed Trade Cards
        lines.append(f"""

{'='*110}
📋 DETAILED TRADE PARAMETERS - TOP 10
{'='*110}
""")
        
        for p in self.positions:
            regime_icon = "🟢" if p.stock_regime == "BULL" else "🔴"
            lines.append(f"""
  {p.ticker} - {p.name} {regime_icon}
  Score: {p.suitability_score:.1f}/100 ({p.recommendation})
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ENTRY PARAMETERS:
    Current Price:      {p.current_price:.2f} SAR
    Target Shares:      {p.target_shares:,} shares
    Position Value:     {p.target_value:,.0f} SAR
    Portfolio Weight:   {p.target_weight_pct:.1f}%
  
  KEY LEVELS:
    50-Day MA:          {p.ma_50:.2f} SAR (Regime change trigger)
    Price vs MA:        {p.price_vs_ma_pct:+.1f}%
    Daily Change:       {p.daily_change_pct:+.2f}%
  
  SCIENTIFIC METRICS:
    Hurst Exponent:     {p.hurst:.3f} {'(Strong Trend)' if p.hurst > 0.55 else '(Weak Trend)'}
    Beta:               {p.beta:.2f} {'(High Beta)' if p.beta > 1.2 else '(Low Beta)'}
    ADX:                {p.adx:.1f} {'(Strong Trend)' if p.adx > 25 else '(Weak Trend)'}
    Volatility:         {p.volatility:.1%} {'(Optimal)' if 0.3 <= p.volatility <= 1.0 else ''}
    20d Momentum:       {p.momentum_20d:+.1%}
""")
        
        # Performance Tracker
        lines.append("=" * 110)
        lines.append("📈 PERFORMANCE TRACKER")
        lines.append("=" * 110)
        
        if self.regime_history:
            bull_days = sum(r.days for r in self.regime_history if r.regime == 'BULL')
            bear_days = sum(r.days for r in self.regime_history if r.regime == 'BEAR')
            total_days = bull_days + bear_days
            
            lines.append(f"""
  Tracking Period:
    Total Days: {total_days}
    Bull Days:  {bull_days} ({bull_days/max(1,total_days)*100:.1f}%)
    Bear Days:  {bear_days} ({bear_days/max(1,total_days)*100:.1f}%)
    Regime Changes: {len(self.regime_history)}
""")
        
        lines.append(f"""

{'='*110}
🎯 ACTION SUMMARY
{'='*110}

  CURRENT RECOMMENDATION:
    Regime: {self.current_regime} → Use {self.current_leverage}x leverage
    {"INCREASE" if self.current_regime == "BULL" else "REDUCE"} exposure to {self.capital * self.current_leverage:,.0f} SAR
    Allocate to {len(self.positions)} scientifically selected stocks
    Monitor: Price crossing {proxy_ma:.2f} SAR triggers regime change

{'='*110}
END OF REPORT
{'='*110}
""")
        
        return "\n".join(lines)
    
    def save_report(self, output_dir: str = "output/production") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.scan_time.strftime('%Y%m%d_%H%M%S')
        filepath = f"{output_dir}/full_market_scan_{timestamp}.txt"
        
        report = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report)
        
        with open(f"{output_dir}/latest_full_market_scan.txt", 'w') as f:
            f.write(report)
        
        return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("FULL TASI MARKET SCANNER")
    print("Scanning ALL stocks to find TOP 10 by scientific score")
    print("Company names fetched directly from Yahoo Finance")
    print("=" * 70 + "\n")
    
    scanner = FullMarketScanner(capital=1_000_000)
    report = scanner.run()
    print(report)
    filepath = scanner.save_report()
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
