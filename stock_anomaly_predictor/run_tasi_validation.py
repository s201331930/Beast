#!/usr/bin/env python3
"""
TASI (Saudi Stock Exchange) Validation Study

Validate the anomaly prediction system on Saudi Arabian stocks.
Saudi stocks use .SR suffix on Yahoo Finance.

This tests the system's effectiveness in a different market:
- Different timezone and trading hours
- Different market dynamics
- Different investor base
- Riyal-denominated assets
"""

import os
import sys
import warnings
import time
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

# Import our modules
from models.stock_screener import StockSuitabilityScreener
from models.statistical_anomaly import StatisticalAnomalyDetector
from models.ml_anomaly import MLAnomalyDetector
from models.cyclical_models import CyclicalAnalyzer
from models.signal_aggregator import SignalAggregator
from analysis.sentiment import SentimentAnalyzer
from backtest.backtester import Backtester, SignalAnalyzer

print("=" * 80)
print("TASI (SAUDI STOCK EXCHANGE) VALIDATION STUDY")
print("Testing Anomaly Prediction System on Saudi Arabian Stocks")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# SAUDI STOCK UNIVERSE
# ============================================================================
# Comprehensive list of Saudi stocks across different sectors
# Format: TICKER.SR for Yahoo Finance

SAUDI_STOCKS = [
    # Oil & Gas / Energy (Saudi Aramco and related)
    '2222.SR',  # Saudi Aramco
    '2030.SR',  # Saudi Kayan Petrochemical
    '2310.SR',  # Sipchem
    '2330.SR',  # Advanced Petrochemical
    '2380.SR',  # Petro Rabigh
    '2060.SR',  # Tasnee
    
    # Banking & Financial Services
    '1180.SR',  # Al Rajhi Bank
    '1010.SR',  # Riyad Bank
    '1050.SR',  # Banque Saudi Fransi
    '1060.SR',  # Samba Financial Group
    '1080.SR',  # Arab National Bank
    '1120.SR',  # Al Jazira Bank
    '1140.SR',  # Bank Al Bilad
    '1150.SR',  # Alinma Bank
    '4280.SR',  # Kingdom Holding
    
    # Materials / Chemicals
    '2010.SR',  # SABIC
    '2020.SR',  # SAFCO
    '2250.SR',  # Saudi Industrial Investment
    '2290.SR',  # Yanbu National Petrochemical
    '3030.SR',  # Saudi Paper Manufacturing
    '2300.SR',  # Saudi Kayan
    
    # Real Estate
    '4300.SR',  # Dar Al Arkan
    '4310.SR',  # Emaar Economic City
    '4020.SR',  # AJIL Financial Services
    '4230.SR',  # Red Sea International
    
    # Retail / Consumer
    '4190.SR',  # Jarir Marketing
    '4001.SR',  # Abdullah Al Othaim Markets
    '4003.SR',  # Extra (United Electronics)
    '4070.SR',  # Tihama Advertising
    '4240.SR',  # Fawaz Alhokair
    '4061.SR',  # Anaam International Holding
    
    # Telecom
    '7010.SR',  # STC (Saudi Telecom)
    '7020.SR',  # Etihad Etisalat (Mobily)
    '7030.SR',  # Zain KSA
    
    # Healthcare
    '4002.SR',  # Mouwasat Medical Services
    '4004.SR',  # Dallah Healthcare
    '4005.SR',  # Care (Medical Care Group)
    '4007.SR',  # Al Hammadi
    '4009.SR',  # Middle East Healthcare
    
    # Food & Beverages
    '2050.SR',  # Savola Group
    '2100.SR',  # Wafrah for Industry
    '2270.SR',  # Saudia Dairy & Foodstuff
    '2280.SR',  # Almarai
    '6001.SR',  # Halwani Bros
    '6002.SR',  # Herfy Food Services
    '6010.SR',  # NADEC
    '6020.SR',  # Aljouf Agriculture
    
    # Industrial / Manufacturing
    '1320.SR',  # Saudi Steel Pipe
    '1301.SR',  # Astra Industrial
    '2040.SR',  # Saudi Ceramic
    '3020.SR',  # Al-Babtain Power
    '3040.SR',  # Saudi Vitrified Clay Pipes
    '3060.SR',  # Saudi Cement
    '3080.SR',  # Eastern Cement
    '3090.SR',  # Saudi Cement
    '3091.SR',  # Al Jouf Cement
    '4031.SR',  # Saudi Airlines Catering
]

# Take 50 random stocks
import random
random.seed(42)  # For reproducibility
SAUDI_STOCKS = random.sample(SAUDI_STOCKS, min(50, len(SAUDI_STOCKS)))

print(f"\nSaudi Stock Universe: {len(SAUDI_STOCKS)} stocks")
print("Sectors: Energy, Banking, Materials, Real Estate, Retail, Telecom, Healthcare, Food, Industrial")

# Analysis parameters
START_DATE = "2021-08-25"
MIN_SCREENING_SCORE = 60

# ============================================================================
# PHASE 1: SCREENING ALL SAUDI STOCKS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: SCREENING SAUDI STOCKS")
print("=" * 80)

# Get market benchmark (using Saudi index or S&P 500 as fallback)
print("\nFetching market benchmark data...")
try:
    # Try TASI index first
    market = yf.Ticker("^TASI")
    market_df = market.history(start=START_DATE)
    if len(market_df) < 100:
        raise ValueError("Insufficient TASI data")
    print("Using TASI index as benchmark")
except:
    # Fallback to S&P 500
    print("TASI index not available, using S&P 500 as benchmark")
    market = yf.Ticker("^GSPC")
    market_df = market.history(start=START_DATE)

market_returns = market_df['Close'].pct_change().dropna()

screening_results = []
failed_tickers = []

for i, ticker in enumerate(SAUDI_STOCKS):
    print(f"\n[{i+1}/{len(SAUDI_STOCKS)}] Screening {ticker}...", end=" ")
    
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE)
        
        if len(df) < 200:
            print(f"Insufficient data ({len(df)} days)")
            failed_tickers.append((ticker, f"Insufficient data ({len(df)} days)"))
            continue
        
        df.columns = [c.lower() for c in df.columns]
        df.name = ticker
        
        # Get company info
        try:
            info = stock.info
            company_name = info.get('shortName', ticker)[:30]
        except:
            company_name = ticker
        
        # Run screening
        screener = StockSuitabilityScreener(df, market_df)
        score = screener.analyze(market_returns)
        
        screening_results.append({
            'ticker': ticker,
            'name': company_name,
            'overall_score': score.overall_score,
            'recommendation': score.recommendation,
            'momentum_score': score.momentum_score,
            'trend_score': score.trend_strength_score,
            'beta_score': score.beta_score,
            'volatility_score': score.volatility_score,
            'liquidity_score': score.liquidity_score,
            'retail_score': score.retail_interest_score,
            'regime_score': score.market_regime_score,
            'hurst': score.hurst_exponent,
            'beta': score.beta,
            'adx': score.adx,
            'volatility': score.volatility,
            'is_trending': score.is_trending,
            'is_high_beta': score.is_high_beta,
            'has_retail_interest': score.has_retail_interest,
            'in_bull_regime': score.in_bull_regime
        })
        
        print(f"{score.overall_score:.1f} ({score.recommendation}) - {company_name}")
        
    except Exception as e:
        print(f"Error: {str(e)[:50]}")
        failed_tickers.append((ticker, str(e)))

# Create screening DataFrame
if not screening_results:
    print("\nERROR: No stocks could be screened successfully!")
    print("This may be due to Yahoo Finance data availability for Saudi stocks.")
    sys.exit(1)

screening_df = pd.DataFrame(screening_results)
screening_df = screening_df.sort_values('overall_score', ascending=False)

print("\n" + "-" * 80)
print("TASI SCREENING SUMMARY")
print("-" * 80)

rec_counts = screening_df['recommendation'].value_counts()
print(f"\nScreened: {len(screening_df)} stocks")
print(f"Failed: {len(failed_tickers)} stocks")
print(f"\nDistribution:")
for rec in ['EXCELLENT', 'GOOD', 'MODERATE', 'POOR']:
    count = rec_counts.get(rec, 0)
    pct = count / len(screening_df) * 100 if len(screening_df) > 0 else 0
    print(f"  {rec}: {count} ({pct:.1f}%)")

# Show top candidates
print("\n" + "-" * 80)
print("TOP 15 SAUDI STOCKS FOR ANALYSIS")
print("-" * 80)
print(f"\n{'Rank':<5} {'Ticker':<12} {'Score':>7} {'Rec':<12} {'Beta':>6} {'Hurst':>7} {'Vol':>7}")
print("-" * 70)

for i, (_, row) in enumerate(screening_df.head(15).iterrows()):
    print(f"{i+1:<5} {row['ticker']:<12} {row['overall_score']:>6.1f} {row['recommendation']:<12} "
          f"{row['beta']:>6.2f} {row['hurst']:>7.3f} {row['volatility']:>6.1%}")

# Filter to GOOD and EXCELLENT only
qualified_df = screening_df[screening_df['overall_score'] >= MIN_SCREENING_SCORE].copy()
print(f"\n\nQualified for full analysis (Score >= {MIN_SCREENING_SCORE}): {len(qualified_df)} stocks")

# Save screening results
screening_df.to_csv('output/tasi_screening_results.csv', index=False)
print("Screening results saved to output/tasi_screening_results.csv")

# ============================================================================
# PHASE 2: FULL ANALYSIS ON QUALIFIED SAUDI STOCKS
# ============================================================================
if len(qualified_df) == 0:
    print("\nNo stocks qualified for full analysis.")
    print("Lowering threshold to 50 for analysis...")
    qualified_df = screening_df[screening_df['overall_score'] >= 50].copy()

print("\n" + "=" * 80)
print("PHASE 2: FULL ANALYSIS ON QUALIFIED SAUDI STOCKS")
print("=" * 80)

analysis_results = []
qualified_tickers = qualified_df['ticker'].tolist()

# Limit to top 20 for time efficiency
if len(qualified_tickers) > 20:
    print(f"\nLimiting analysis to top 20 stocks (from {len(qualified_tickers)})")
    qualified_tickers = qualified_tickers[:20]

print(f"\nRunning full analysis on {len(qualified_tickers)} Saudi stocks...")

for i, ticker in enumerate(qualified_tickers):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(qualified_tickers)}] ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    try:
        # Get screening score
        screen_row = qualified_df[qualified_df['ticker'] == ticker].iloc[0]
        screen_score = screen_row['overall_score']
        company_name = screen_row['name']
        
        print(f"  Company: {company_name}")
        
        # Fetch data
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE)
        df.columns = [c.lower() for c in df.columns]
        df['returns'] = df['close'].pct_change()
        
        print(f"  Data: {len(df)} days, Price: {df['close'].iloc[-1]:.2f} SAR")
        
        # Statistical Anomaly Detection
        print("  Running statistical anomaly detection...")
        stat_detector = StatisticalAnomalyDetector(df)
        stat_anomalies = stat_detector.run_all_detectors()
        
        # ML Anomaly Detection
        print("  Running ML anomaly detection...")
        ml_detector = MLAnomalyDetector(df)
        ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)
        
        # Cyclical Analysis
        print("  Running cyclical analysis...")
        cyclical_analyzer = CyclicalAnalyzer(df)
        cyclical_signals = cyclical_analyzer.run_all_analysis()
        
        # Sentiment
        print("  Running sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer(ticker)
        sentiment_signals = sentiment_analyzer.run_full_analysis(df)
        
        # Signal Aggregation
        print("  Aggregating signals...")
        aggregator = SignalAggregator()
        aggregator.merge_all_signals(
            df,
            stat_anomalies,
            ml_anomalies,
            cyclical_signals,
            sentiment_signals,
            pd.DataFrame(index=df.index)
        )
        
        trade_signals = aggregator.generate_trade_signals(
            prob_threshold=0.55,
            confidence_threshold=0.3,
            anomaly_threshold=0.4
        )
        
        # Backtesting
        print("  Running backtest...")
        backtester = Backtester()
        backtest_result = backtester.run_backtest(
            df,
            trade_signals,
            signal_column='actionable_signal'
        )
        
        # Signal analysis
        signal_analyzer = SignalAnalyzer(trade_signals, df['returns'])
        hit_rate = signal_analyzer.analyze_signal_hit_rate('actionable_signal')
        magnitude = signal_analyzer.analyze_signal_magnitude('actionable_signal')
        
        # Store results
        analysis_results.append({
            'ticker': ticker,
            'name': company_name,
            'screening_score': screen_score,
            'recommendation': screen_row['recommendation'],
            'beta': screen_row['beta'],
            'hurst': screen_row['hurst'],
            'current_price': df['close'].iloc[-1],
            'total_trades': backtest_result.total_trades,
            'win_rate': backtest_result.win_rate,
            'profit_factor': backtest_result.profit_factor,
            'total_return': backtest_result.total_return,
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'max_drawdown': backtest_result.max_drawdown,
            'hit_rate_5d': hit_rate.get('5d_hit_rate', 0),
            'hit_rate_10d': hit_rate.get('10d_hit_rate', 0),
            'hit_rate_20d': hit_rate.get('20d_hit_rate', 0),
            'avg_signal_return': magnitude.get('avg_signal_return', 0),
            'lift': magnitude.get('lift', 0)
        })
        
        print(f"  Results: {backtest_result.total_trades} trades, "
              f"{backtest_result.win_rate:.1%} win rate, "
              f"PF: {backtest_result.profit_factor:.2f}")
        
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        analysis_results.append({
            'ticker': ticker,
            'name': screen_row.get('name', ticker),
            'screening_score': screen_row['overall_score'],
            'recommendation': screen_row['recommendation'],
            'beta': screen_row['beta'],
            'hurst': screen_row['hurst'],
            'current_price': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'hit_rate_5d': 0,
            'hit_rate_10d': 0,
            'hit_rate_20d': 0,
            'avg_signal_return': 0,
            'lift': 0,
            'error': str(e)
        })

# Create analysis DataFrame
analysis_df = pd.DataFrame(analysis_results)
analysis_df = analysis_df.sort_values('screening_score', ascending=False)

# ============================================================================
# PHASE 3: STATISTICAL VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: STATISTICAL VALIDATION - TASI")
print("=" * 80)

# Filter to stocks with sufficient trades
valid_analysis = analysis_df[analysis_df['total_trades'] >= 5].copy()

print(f"\nStocks with ≥5 trades: {len(valid_analysis)}")

correlations = {}

if len(valid_analysis) >= 3:
    print("\n" + "-" * 80)
    print("CORRELATION ANALYSIS: Screening Score vs Performance")
    print("-" * 80)
    
    metrics = ['win_rate', 'profit_factor', 'hit_rate_10d', 'avg_signal_return']
    
    for metric in metrics:
        if metric in valid_analysis.columns:
            valid_data = valid_analysis[['screening_score', metric]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_data) >= 3:
                corr, p_value = stats.pearsonr(valid_data['screening_score'], valid_data[metric])
                correlations[metric] = {'correlation': corr, 'p_value': p_value}
                
                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"  {metric:<20}: r = {corr:+.3f}, p = {p_value:.4f} {sig}")

# Performance by category
print("\n" + "-" * 80)
print("PERFORMANCE BY SCREENING CATEGORY (TASI)")
print("-" * 80)

for rec in ['EXCELLENT', 'GOOD', 'MODERATE']:
    group = valid_analysis[valid_analysis['recommendation'] == rec]
    if len(group) > 0:
        print(f"\n{rec} ({len(group)} stocks):")
        print(f"  Avg Win Rate:      {group['win_rate'].mean():.1%}")
        print(f"  Avg Profit Factor: {group['profit_factor'].mean():.2f}")
        print(f"  Avg 10D Hit Rate:  {group['hit_rate_10d'].mean():.1%}")

# Top performers
print("\n" + "-" * 80)
print("TOP 10 SAUDI PERFORMERS (by 10-Day Hit Rate)")
print("-" * 80)

top_performers = analysis_df[analysis_df['total_trades'] >= 5].nlargest(10, 'hit_rate_10d')
print(f"\n{'Ticker':<12} {'Name':<20} {'Screen':>7} {'WinRate':>8} {'10D Hit':>8} {'PF':>6}")
print("-" * 70)
for _, row in top_performers.iterrows():
    name = str(row['name'])[:18] if pd.notna(row['name']) else row['ticker']
    print(f"{row['ticker']:<12} {name:<20} {row['screening_score']:>6.1f} "
          f"{row['win_rate']:>7.1%} {row['hit_rate_10d']:>7.1%} {row['profit_factor']:>6.2f}")

# ============================================================================
# COMPARISON: TASI vs US MARKET
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TASI vs US MARKET")
print("=" * 80)

# US market results (from previous study)
us_results = {
    'avg_win_rate': 0.459,
    'avg_profit_factor': 1.18,
    'avg_hit_rate_10d': 0.533,
    'correlation_win_rate': 0.399
}

if len(valid_analysis) >= 3:
    tasi_avg_win_rate = valid_analysis['win_rate'].mean()
    tasi_avg_pf = valid_analysis['profit_factor'].mean()
    tasi_avg_hit_10d = valid_analysis['hit_rate_10d'].mean()
    tasi_corr = correlations.get('win_rate', {}).get('correlation', 0)
    
    print(f"""
                                    TASI          US Market
    ───────────────────────────────────────────────────────────
    Avg Win Rate:               {tasi_avg_win_rate:>6.1%}         {us_results['avg_win_rate']:.1%}
    Avg Profit Factor:          {tasi_avg_pf:>6.2f}          {us_results['avg_profit_factor']:.2f}
    Avg 10D Hit Rate:           {tasi_avg_hit_10d:>6.1%}         {us_results['avg_hit_rate_10d']:.1%}
    Screen-WinRate Correlation: {tasi_corr:>+6.3f}         {us_results['correlation_win_rate']:+.3f}
    """)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TASI VALIDATION FINAL SUMMARY")
print("=" * 80)

print(f"""
STUDY OVERVIEW:
───────────────────────────────────────────────────────────────────────────────
  Market:                    TASI (Saudi Stock Exchange)
  Stocks Screened:           {len(screening_df)}
  Stocks Analyzed:           {len(analysis_df)}
  With Sufficient Trades:    {len(valid_analysis)}
  Analysis Period:           {START_DATE} to present
""")

if len(valid_analysis) >= 3:
    # High vs Low score comparison
    median_score = valid_analysis['screening_score'].median()
    high_score = valid_analysis[valid_analysis['screening_score'] >= median_score]
    low_score = valid_analysis[valid_analysis['screening_score'] < median_score]
    
    if len(high_score) > 0 and len(low_score) > 0:
        print(f"""SCREENING EFFECTIVENESS (TASI):
───────────────────────────────────────────────────────────────────────────────

  HIGH SCORE STOCKS (≥{median_score:.1f}):
    Count:           {len(high_score)}
    Avg Win Rate:    {high_score['win_rate'].mean():.1%}
    Avg 10D Hit:     {high_score['hit_rate_10d'].mean():.1%}
    Avg PF:          {high_score['profit_factor'].mean():.2f}

  LOW SCORE STOCKS (<{median_score:.1f}):
    Count:           {len(low_score)}
    Avg Win Rate:    {low_score['win_rate'].mean():.1%}
    Avg 10D Hit:     {low_score['hit_rate_10d'].mean():.1%}
    Avg PF:          {low_score['profit_factor'].mean():.2f}
""")

# Best candidates
print("""RECOMMENDED SAUDI STOCKS FOR TRADING:
───────────────────────────────────────────────────────────────────────────────
""")

best_candidates = analysis_df[
    (analysis_df['total_trades'] >= 5) &
    (analysis_df['hit_rate_10d'] >= 0.50) &
    (analysis_df['profit_factor'] >= 1.0)
].sort_values('hit_rate_10d', ascending=False)

if len(best_candidates) > 0:
    print("  Top TASI candidates (10D Hit ≥50%, PF ≥1.0):\n")
    for _, row in best_candidates.head(10).iterrows():
        name = str(row['name'])[:20] if pd.notna(row['name']) else ''
        print(f"    ✓ {row['ticker']}: {name}")
        print(f"      Screen={row['screening_score']:.1f}, 10D Hit={row['hit_rate_10d']:.1%}, PF={row['profit_factor']:.2f}")
else:
    print("  No stocks met all criteria in this validation.")

print(f"""
CONCLUSION:
───────────────────────────────────────────────────────────────────────────────
""")

if len(valid_analysis) >= 3 and 'win_rate' in correlations:
    corr = correlations['win_rate']['correlation']
    pval = correlations['win_rate']['p_value']
    
    if corr > 0.3 and pval < 0.1:
        verdict = "VALIDATED FOR TASI ✓"
        explanation = "System shows positive correlation with performance in Saudi market"
    elif corr > 0:
        verdict = "PARTIALLY VALIDATED"
        explanation = "Positive but weaker correlation observed in TASI"
    else:
        verdict = "DIFFERENT DYNAMICS"
        explanation = "Saudi market may have different characteristics affecting signals"
    
    print(f"  Status: {verdict}")
    print(f"  {explanation}")
    print(f"  Correlation (Screen Score ↔ Win Rate): r = {corr:.3f}, p = {pval:.4f}")
else:
    print("  Insufficient data for conclusive validation")

print("\n" + "=" * 80)
print("TASI VALIDATION COMPLETE")
print("=" * 80)

# Save results
analysis_df.to_csv('output/tasi_analysis_results.csv', index=False)
print("\nResults saved to output/tasi_analysis_results.csv")

all_results = {
    'screening_df': screening_df,
    'analysis_df': analysis_df,
    'correlations': correlations,
    'timestamp': datetime.now().isoformat(),
    'market': 'TASI'
}

with open('output/tasi_validation_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("Full results saved to output/tasi_validation_results.pkl")
