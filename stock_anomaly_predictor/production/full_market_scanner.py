#!/usr/bin/env python3
"""
FULL MARKET SCANNER
===================
Scans ALL TASI (Saudi Stock Exchange) stocks and selects the TOP 10
based on scientific screening criteria.

Complete TASI stock list with all sectors.
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
# COMPLETE TASI STOCK LIST (All Sectors)
# =============================================================================

# This is a comprehensive list of TASI stocks across all sectors
ALL_TASI_STOCKS = {
    # === BANKS ===
    '1180.SR': 'Al Rajhi Bank',
    '1010.SR': 'Riyad Bank',
    '1150.SR': 'Alinma Bank',
    '1140.SR': 'Banque Saudi Fransi',
    '1060.SR': 'Saudi Investment Bank',
    '1050.SR': 'Saudi British Bank (SABB)',
    '1020.SR': 'Bank Al-Jazira',
    '1030.SR': 'Saudi Awwal Bank (SAB)',
    '1080.SR': 'Arab National Bank',
    '1182.SR': 'Amlak International',
    
    # === ENERGY ===
    '2222.SR': 'Saudi Aramco',
    '2030.SR': 'Sarco',
    '4030.SR': 'Bahri',
    '2381.SR': 'Petro Rabigh',
    '2380.SR': 'Rabigh Refining',
    
    # === MATERIALS ===
    '2010.SR': 'SABIC',
    '1211.SR': 'Maaden',
    '2290.SR': 'Yanbu National Petrochemical',
    '2250.SR': 'Saudi Industrial Investment',
    '2210.SR': 'Nama Chemicals',
    '2001.SR': 'Methanol Chemicals',
    '2060.SR': 'National Industrialization',
    '2310.SR': 'Saudi International Petrochemical',
    '2350.SR': 'Saudi Kayan',
    '2330.SR': 'Advanced Petrochemical',
    '2170.SR': 'Alujain Corporation',
    '2020.SR': 'Saudi Arabia Fertilizers (SAFCO)',
    '1320.SR': 'Saudi Steel Pipe',
    '1321.SR': 'Astra Industrial Group',
    '1302.SR': 'Bawan Company',
    '1304.SR': 'Al Yamamah Steel',
    '2200.SR': 'Arabian Pipe',
    '2220.SR': 'National Metal Manufacturing',
    '3010.SR': 'Arabian Cement',
    '3020.SR': 'Yamama Cement',
    '3030.SR': 'Saudi Cement',
    '3040.SR': 'Qassim Cement',
    '3050.SR': 'Southern Province Cement',
    '3060.SR': 'Yanbu Cement',
    '3080.SR': 'Eastern Province Cement',
    '3090.SR': 'Tabuk Cement',
    '3091.SR': 'Umm Al-Qura Cement',
    '3001.SR': 'Hail Cement',
    '3002.SR': 'Najran Cement',
    '3003.SR': 'City Cement',
    '3004.SR': 'Northern Region Cement',
    '2190.SR': 'Sipchem',
    '2240.SR': 'Zamil Industrial',
    
    # === INDUSTRIALS ===
    '2082.SR': 'ACWA Power',
    '2083.SR': 'Alkhorayef Water & Power',
    '1212.SR': 'Astra Industrial',
    '2040.SR': 'Saudi Ceramic',
    '2130.SR': 'Saudi Industrial Development',
    '2320.SR': 'Albabtain Power',
    '2370.SR': 'Middle East Specialized Cables',
    '4110.SR': 'Saudi Printing & Packaging',
    '4140.SR': 'Saudi Industrial Services',
    '2160.SR': 'Al Hassan Shaker (Jokey)',
    '1214.SR': 'Al Sorayai Trading',
    '2180.SR': 'Filling & Packing Materials (FIPCO)',
    '2150.SR': 'Saudi Industrial Export Company',
    '4142.SR': 'Al Kathiri Holding',
    '4143.SR': 'Alujain Corporation',
    
    # === CONSUMER DISCRETIONARY ===
    '4190.SR': 'Jarir Marketing',
    '4003.SR': 'Extra',
    '4240.SR': 'Fawaz Abdulaziz Alhokair',
    '4001.SR': 'Abdullah Al Othaim Markets',
    '4002.SR': 'Mouwasat Medical Services',
    '4004.SR': 'Dallah Healthcare',
    '4005.SR': 'Saudi Pharmaceutical Industries',
    '4006.SR': 'Saudi Arabian Cooperative Insurance',
    '4007.SR': 'Al Hammadi Holding',
    '4008.SR': 'National Medical Care',
    '4009.SR': 'SACO',
    '4010.SR': 'Saudi Paper Manufacturing',
    '4011.SR': 'Al Sagr Cooperative Insurance',
    '4012.SR': 'Tihama Advertising',
    '4013.SR': 'Walaa Cooperative Insurance',
    '4014.SR': 'Al Alamiya Insurance',
    '4015.SR': 'Fitaihi Holding',
    '4017.SR': 'Saudi Hotels',
    '4020.SR': 'Saudi Public Transport',
    '4031.SR': 'Saudi Ground Services',
    '4040.SR': 'Saudi Real Estate',
    '4050.SR': 'Saudi Automotive Services',
    '4051.SR': 'Taiba Holding',
    '4061.SR': 'Anaam International Holding',
    '4071.SR': 'Almarai',
    '4080.SR': 'Saudia Dairy & Foodstuff (SADAFCO)',
    '4130.SR': 'Al Babtain Power & Telecom',
    '4160.SR': 'Thimar Development',
    '4180.SR': 'Fitaihi Holding',
    '4200.SR': 'Aldrees Petroleum',
    '4210.SR': 'Saudi Automotive Services',
    '4220.SR': 'Emaar The Economic City',
    '4230.SR': 'Red Sea International',
    '4250.SR': 'Jabal Omar',
    '4260.SR': 'Budget Saudi',
    '4261.SR': 'Mohammad Al Mojil Group (MMG)',
    '4270.SR': 'Saudi Research & Marketing Group',
    '4280.SR': 'Kingdom Holding Company',
    '4290.SR': 'Al Khaleej Training & Education',
    '4291.SR': 'NCLE',
    '4292.SR': 'Sasco',
    
    # === CONSUMER STAPLES ===
    '2280.SR': 'Almarai',
    '2050.SR': 'Savola Group',
    '6002.SR': 'Herfy Food Services',
    '6001.SR': 'Halwani Bros',
    '6010.SR': 'National Agricultural Development (NADEC)',
    '6020.SR': 'Jazan Development Company',
    '6040.SR': 'Tabuk Agricultural Development',
    '6050.SR': 'Saudi Fisheries',
    '6060.SR': 'Ash-Sharqiyah Development',
    '6070.SR': 'Al Jouf Agricultural Development',
    '6090.SR': 'Jadwa REIT Saudi Fund',
    '4291.SR': 'National Company for Learning & Education',
    
    # === HEALTHCARE ===
    '4002.SR': 'Mouwasat Medical Services',
    '4004.SR': 'Dallah Healthcare',
    '4007.SR': 'Al Hammadi Holding',
    '4009.SR': 'Saudi Chemical Company',
    
    # === TELECOM ===
    '7010.SR': 'STC (Saudi Telecom)',
    '7020.SR': 'Mobily (Etihad Etisalat)',
    '7030.SR': 'Zain KSA',
    '7040.SR': 'Integrated Telecom',
    
    # === UTILITIES ===
    '5110.SR': 'Saudi Electricity Company',
    '2082.SR': 'ACWA Power',
    '2083.SR': 'Alkhorayef Water & Power Technologies',
    
    # === REAL ESTATE ===
    '4300.SR': 'Dar Al Arkan Real Estate',
    '4310.SR': 'Knowledge Economic City',
    '4320.SR': 'Al Andalus Property',
    '4321.SR': 'Riyad REIT',
    '4322.SR': 'Jadwa REIT Al Haramain Fund',
    '4323.SR': 'SEDCO Capital REIT Fund',
    '4324.SR': 'Bonyan REIT Fund',
    '4330.SR': 'Arriyadh Development Company',
    '4331.SR': 'Makkah Construction & Development',
    '4332.SR': 'Jabal Omar Development',
    '4333.SR': 'Taiba Holding',
    '4334.SR': 'Al Tayyar Travel Group',
    '4336.SR': 'Alinma Retail REIT',
    '4337.SR': 'Jazira REIT Fund',
    '4338.SR': 'Al Rajhi REIT Fund',
    '4339.SR': 'Derayah REIT Fund',
    '4340.SR': 'Swicorp Wabel REIT Fund',
    '4342.SR': 'Al Maather REIT',
    '4344.SR': 'Mulkia Gulf Real Estate REIT',
    '4345.SR': 'Saudi Enaya Cooperative Insurance',
    '4347.SR': 'Musharaka REIT',
    
    # === INSURANCE ===
    '8010.SR': 'Tawuniya',
    '8012.SR': 'Bupa Arabia',
    '8020.SR': 'Malath Cooperative Insurance',
    '8030.SR': 'Mediterranean & Gulf Insurance',
    '8040.SR': 'Allianz Saudi Fransi Cooperative Insurance',
    '8050.SR': 'Salama Cooperative Insurance',
    '8060.SR': 'Walaa Cooperative Insurance',
    '8070.SR': 'Arabian Shield Insurance',
    '8080.SR': 'SABB Takaful',
    '8100.SR': 'Saudi Re for Cooperative Reinsurance',
    '8120.SR': 'Gulf Union Cooperative Insurance',
    '8150.SR': 'ACIG',
    '8160.SR': 'Al Ahlia For Cooperative Insurance',
    '8170.SR': 'Al Ahli Takaful Company',
    '8180.SR': 'Al Sagr Cooperative Insurance',
    '8190.SR': 'United Cooperative Assurance',
    '8200.SR': 'Al Rajhi Takaful',
    '8210.SR': 'Bupa Arabia for Cooperative Insurance',
    '8230.SR': 'Buruj Cooperative Insurance',
    '8240.SR': 'Axa Cooperative Insurance',
    '8250.SR': 'Trade Union Cooperative Insurance',
    '8260.SR': 'Gulf General Cooperative Insurance',
    '8270.SR': 'Arabian Cooperative Insurance',
    '8280.SR': 'Al Alamiya Insurance',
    '8300.SR': 'Wataniya Insurance',
    '8310.SR': 'Amana Cooperative Insurance',
    '8311.SR': 'Aljazira Takaful',
    
    # === REITS ===
    '1120.SR': 'Al Rajhi REIT',
    '4330.SR': 'Arriyadh Development Company',
    '4336.SR': 'Alinma Retail REIT',
    '4337.SR': 'Jazira REIT',
    '4338.SR': 'Al Rajhi REIT Fund',
    '4339.SR': 'Derayah REIT',
    '4340.SR': 'Swicorp Wabel REIT',
    '4342.SR': 'Al Maather REIT',
    '4344.SR': 'Mulkia Gulf REIT',
    '4345.SR': 'Enaya Insurance',
    '4347.SR': 'Musharaka REIT',
    
    # === DIVERSIFIED FINANCIALS ===
    '1111.SR': 'Saudi Tadawul Group',
    '4280.SR': 'Kingdom Holding',
    '4081.SR': 'Nat\'l Agriculture Company',
    '1183.SR': 'Saudi Arabian Mining',
    '4082.SR': 'National Agricultural Marketing',
}

# Remove duplicates and invalid entries
ALL_TASI_STOCKS = {k: v for k, v in ALL_TASI_STOCKS.items() if k and v}


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
    name: str
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
        """Fetch data for ALL TASI stocks."""
        total = len(ALL_TASI_STOCKS)
        print(f"Fetching data for {total} TASI stocks...")
        print("This may take a few minutes...\n")
        
        loaded = 0
        failed = 0
        
        for i, (ticker, name) in enumerate(ALL_TASI_STOCKS.items()):
            try:
                df = yf.Ticker(ticker).history(period="1y")
                if len(df) >= 60:
                    df.columns = [c.lower() for c in df.columns]
                    self.stock_data[ticker] = df
                    loaded += 1
                else:
                    failed += 1
            except:
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
            name = ALL_TASI_STOCKS.get(ticker, ticker)
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
        lines.append("=" * 110)
        
        lines.append(f"""
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FULL MARKET SCREENING SUMMARY                                                                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  Total TASI Stocks in Database:  {len(ALL_TASI_STOCKS):>5}                                                                       │
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
        
        lines.append(f"\n  {'Rank':<5} {'Ticker':<10} {'Name':<25} {'Score':>7} {'Rec':<10} {'Hurst':>7} {'Beta':>6} {'ADX':>6} {'Vol':>7} {'Mom20d':>8}")
        lines.append("  " + "-" * 105)
        
        for i, score in enumerate(self.all_scores[:20]):
            selected = "✓" if score.ticker in self.selected_stocks else " "
            lines.append(f"  {i+1:>3}{selected} {score.ticker:<10} {score.name[:25]:<25} {score.overall_score:>6.1f} "
                        f"{score.recommendation:<10} {score.hurst_exponent:>7.3f} {score.beta:>6.2f} {score.adx:>6.1f} "
                        f"{score.volatility:>6.1%} {score.momentum_20d:>+7.1%}")
        
        # Current Regime
        regime_emoji = "🟢" if self.current_regime == "BULL" else "🔴"
        proxy = self.selected_stocks[0] if self.selected_stocks else "N/A"
        proxy_price = self.stock_data[proxy]['close'].iloc[-1] if proxy in self.stock_data else 0
        proxy_ma = self.stock_data[proxy]['close'].rolling(50).mean().iloc[-1] if proxy in self.stock_data else 0
        
        lines.append(f"""

{'='*110}
{regime_emoji} CURRENT MARKET REGIME: {self.current_regime}
{'='*110}

  Market Proxy ({proxy}):
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
        
        lines.append(f"  {'Ticker':<10} {'Name':<22} {'Score':>6} {'Price':>10} {'50-MA':>10} {'vs MA':>8} {'Weight':>8} {'Shares':>8} {'Value':>12} {'Regime':<6}")
        lines.append("  " + "-" * 115)
        
        total_value = 0
        for p in self.positions:
            regime_icon = "🟢" if p.stock_regime == "BULL" else "🔴"
            lines.append(f"  {p.ticker:<10} {p.name[:22]:<22} {p.suitability_score:>5.1f} {p.current_price:>10.2f} "
                        f"{p.ma_50:>10.2f} {p.price_vs_ma_pct:>+7.1f}% {p.target_weight_pct:>7.1f}% "
                        f"{p.target_shares:>8} {p.target_value:>12,.0f} {regime_icon}")
            total_value += p.target_value
        
        lines.append("  " + "-" * 115)
        total_weight = sum(p.target_weight_pct for p in self.positions)
        lines.append(f"  {'TOTAL':<10} {'':<22} {'':<6} {'':<10} {'':<10} {'':<8} {total_weight:>7.1f}% {'':<8} {total_value:>12,.0f}")
        
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
    print("=" * 70 + "\n")
    
    scanner = FullMarketScanner(capital=1_000_000)
    report = scanner.run()
    print(report)
    filepath = scanner.save_report()
    print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
