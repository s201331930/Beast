#!/usr/bin/env python3
"""
Large-Scale Validation Study

Scientific validation of the screening system using 50 diverse stocks.
Methodology:
1. Screen all 50 stocks
2. Classify by screening score (EXCELLENT, GOOD, MODERATE, POOR)
3. Run full analysis on GOOD+ candidates
4. Validate correlation between screening score and backtest performance
5. Statistical significance testing

This follows rigorous scientific methodology to validate our hypothesis:
"Higher screening scores predict better anomaly detection performance"
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
from analysis.market_context import MarketContextAnalyzer
from backtest.backtester import Backtester, SignalAnalyzer
from config.settings import config

print("=" * 80)
print("LARGE-SCALE VALIDATION STUDY")
print("Testing Anomaly Prediction System on 50 Diverse Stocks")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# STOCK UNIVERSE SELECTION
# ============================================================================
# Diverse selection across:
# - Market cap (Large, Mid, Small)
# - Sectors (Tech, Healthcare, Finance, Consumer, Energy, etc.)
# - Volatility profiles (High, Medium, Low)
# - Retail interest (Meme stocks, ETFs, Blue chips)

STOCK_UNIVERSE = [
    # Large Cap Tech (High retail interest)
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'NVDA', 'AMD',
    
    # Semiconductors
    'AVGO', 'QCOM', 'MU', 'INTC', 'MRVL', 'AMAT',
    
    # High Beta / Momentum
    'COIN', 'MARA', 'RIOT', 'SQ', 'SHOP', 'SNOW',
    
    # Meme / Retail Favorites
    'GME', 'AMC', 'PLTR', 'SOFI', 'HOOD', 'RIVN',
    
    # Space / Defense (similar to RKLB)
    'SPCE', 'LMT', 'NOC', 'RTX', 'BA',
    
    # Biotech / Healthcare
    'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA',
    
    # Finance
    'JPM', 'GS', 'MS', 'C', 'BAC',
    
    # Consumer
    'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX',
    
    # Energy
    'XOM', 'CVX', 'OXY', 'DVN', 'FANG',
    
    # Chinese ADRs (like BABA)
    'BABA', 'JD', 'PDD', 'BIDU', 'NIO'
]

# Ensure we have exactly 50
STOCK_UNIVERSE = STOCK_UNIVERSE[:50]
print(f"\nStock Universe: {len(STOCK_UNIVERSE)} stocks")

# Analysis parameters
START_DATE = "2021-08-25"
MIN_SCREENING_SCORE = 60  # GOOD or EXCELLENT threshold

# ============================================================================
# PHASE 1: SCREENING ALL STOCKS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: SCREENING ALL 50 STOCKS")
print("=" * 80)

# Get market data for beta calculation
print("\nFetching market benchmark data...")
market = yf.Ticker("^GSPC")
market_df = market.history(start=START_DATE)
market_returns = market_df['Close'].pct_change().dropna()

screening_results = []
failed_tickers = []

for i, ticker in enumerate(STOCK_UNIVERSE):
    print(f"\n[{i+1}/{len(STOCK_UNIVERSE)}] Screening {ticker}...", end=" ")
    
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE)
        
        if len(df) < 200:
            print(f"Insufficient data ({len(df)} days)")
            failed_tickers.append((ticker, "Insufficient data"))
            continue
        
        df.columns = [c.lower() for c in df.columns]
        df.name = ticker
        
        # Run screening
        screener = StockSuitabilityScreener(df, market_df)
        score = screener.analyze(market_returns)
        
        screening_results.append({
            'ticker': ticker,
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
        
        print(f"Score: {score.overall_score:.1f} ({score.recommendation})")
        
    except Exception as e:
        print(f"Error: {str(e)[:50]}")
        failed_tickers.append((ticker, str(e)))

# Create screening DataFrame
screening_df = pd.DataFrame(screening_results)
screening_df = screening_df.sort_values('overall_score', ascending=False)

print("\n" + "-" * 80)
print("SCREENING SUMMARY")
print("-" * 80)

# Count by recommendation
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
print("TOP 15 CANDIDATES FOR ANALYSIS")
print("-" * 80)
print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':>7} {'Rec':<12} {'Beta':>6} {'Hurst':>7} {'Vol':>7}")
print("-" * 60)

for i, (_, row) in enumerate(screening_df.head(15).iterrows()):
    print(f"{i+1:<5} {row['ticker']:<8} {row['overall_score']:>6.1f} {row['recommendation']:<12} "
          f"{row['beta']:>6.2f} {row['hurst']:>7.3f} {row['volatility']:>6.1%}")

# Filter to GOOD and EXCELLENT only
qualified_df = screening_df[screening_df['overall_score'] >= MIN_SCREENING_SCORE].copy()
print(f"\n\nQualified for full analysis (Score >= {MIN_SCREENING_SCORE}): {len(qualified_df)} stocks")

# Save screening results
screening_df.to_csv('output/screening_all_stocks.csv', index=False)
print("Screening results saved to output/screening_all_stocks.csv")

# ============================================================================
# PHASE 2: FULL ANALYSIS ON QUALIFIED STOCKS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: FULL ANALYSIS ON QUALIFIED STOCKS")
print("=" * 80)

analysis_results = []
qualified_tickers = qualified_df['ticker'].tolist()

print(f"\nRunning full analysis on {len(qualified_tickers)} qualified stocks...")
print("This may take several minutes...\n")

for i, ticker in enumerate(qualified_tickers):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(qualified_tickers)}] ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    try:
        # Get screening score for this ticker
        screen_row = qualified_df[qualified_df['ticker'] == ticker].iloc[0]
        screen_score = screen_row['overall_score']
        
        # Fetch fresh data
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE)
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        print(f"  Data: {len(df)} days")
        
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
        
        # Sentiment (simulated)
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
            pd.DataFrame(index=df.index)  # Empty market signals for speed
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
            'screening_score': screen_score,
            'recommendation': screen_row['recommendation'],
            'beta': screen_row['beta'],
            'hurst': screen_row['hurst'],
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
            'avg_nonsignal_return': magnitude.get('avg_non_signal_return', 0),
            'lift': magnitude.get('lift', 0)
        })
        
        print(f"  Results: {backtest_result.total_trades} trades, "
              f"{backtest_result.win_rate:.1%} win rate, "
              f"PF: {backtest_result.profit_factor:.2f}")
        
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        analysis_results.append({
            'ticker': ticker,
            'screening_score': screen_row['overall_score'],
            'recommendation': screen_row['recommendation'],
            'beta': screen_row['beta'],
            'hurst': screen_row['hurst'],
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
            'avg_nonsignal_return': 0,
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
print("PHASE 3: STATISTICAL VALIDATION")
print("=" * 80)

# Filter to stocks with sufficient trades
valid_analysis = analysis_df[analysis_df['total_trades'] >= 10].copy()

print(f"\nStocks with ≥10 trades: {len(valid_analysis)}")

if len(valid_analysis) >= 5:
    # Calculate correlations
    correlations = {}
    
    metrics = ['win_rate', 'profit_factor', 'hit_rate_10d', 'avg_signal_return']
    
    print("\n" + "-" * 80)
    print("CORRELATION ANALYSIS: Screening Score vs Performance")
    print("-" * 80)
    
    for metric in metrics:
        if metric in valid_analysis.columns:
            # Remove any inf/nan
            valid_data = valid_analysis[['screening_score', metric]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(valid_data) >= 3:
                corr, p_value = stats.pearsonr(valid_data['screening_score'], valid_data[metric])
                correlations[metric] = {'correlation': corr, 'p_value': p_value}
                
                sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                print(f"  {metric:<20}: r = {corr:+.3f}, p = {p_value:.4f} {sig}")

    # Group analysis by recommendation
    print("\n" + "-" * 80)
    print("PERFORMANCE BY SCREENING CATEGORY")
    print("-" * 80)
    
    for rec in ['EXCELLENT', 'GOOD']:
        group = valid_analysis[valid_analysis['recommendation'] == rec]
        if len(group) > 0:
            print(f"\n{rec} ({len(group)} stocks):")
            print(f"  Avg Win Rate:      {group['win_rate'].mean():.1%}")
            print(f"  Avg Profit Factor: {group['profit_factor'].mean():.2f}")
            print(f"  Avg 10D Hit Rate:  {group['hit_rate_10d'].mean():.1%}")
            print(f"  Avg Total Return:  {group['total_return'].mean():.2%}")

# Top performers
print("\n" + "-" * 80)
print("TOP 10 PERFORMERS (by 10-Day Hit Rate)")
print("-" * 80)

top_performers = analysis_df[analysis_df['total_trades'] >= 5].nlargest(10, 'hit_rate_10d')
print(f"\n{'Ticker':<8} {'Screen':>8} {'Trades':>7} {'WinRate':>8} {'10D Hit':>8} {'PF':>6}")
print("-" * 55)
for _, row in top_performers.iterrows():
    print(f"{row['ticker']:<8} {row['screening_score']:>7.1f} {row['total_trades']:>7} "
          f"{row['win_rate']:>7.1%} {row['hit_rate_10d']:>7.1%} {row['profit_factor']:>6.2f}")

# Bottom performers (for comparison)
print("\n" + "-" * 80)
print("BOTTOM 10 PERFORMERS (by 10-Day Hit Rate)")
print("-" * 80)

bottom_performers = analysis_df[analysis_df['total_trades'] >= 5].nsmallest(10, 'hit_rate_10d')
print(f"\n{'Ticker':<8} {'Screen':>8} {'Trades':>7} {'WinRate':>8} {'10D Hit':>8} {'PF':>6}")
print("-" * 55)
for _, row in bottom_performers.iterrows():
    print(f"{row['ticker']:<8} {row['screening_score']:>7.1f} {row['total_trades']:>7} "
          f"{row['win_rate']:>7.1%} {row['hit_rate_10d']:>7.1%} {row['profit_factor']:>6.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

# Calculate aggregate statistics
excellent_stocks = analysis_df[analysis_df['recommendation'] == 'EXCELLENT']
good_stocks = analysis_df[analysis_df['recommendation'] == 'GOOD']

print(f"""
STUDY OVERVIEW:
───────────────────────────────────────────────────────────────────────────────
  Total stocks screened:     {len(screening_df)}
  Passed screening (≥60):    {len(qualified_df)} ({len(qualified_df)/len(screening_df)*100:.1f}%)
  Successfully analyzed:     {len(analysis_df[analysis_df['total_trades'] > 0])}
  With sufficient trades:    {len(valid_analysis)}

SCREENING EFFECTIVENESS:
───────────────────────────────────────────────────────────────────────────────
""")

if len(valid_analysis) >= 3:
    # Calculate if higher scores = better performance
    median_score = valid_analysis['screening_score'].median()
    high_score = valid_analysis[valid_analysis['screening_score'] >= median_score]
    low_score = valid_analysis[valid_analysis['screening_score'] < median_score]
    
    if len(high_score) > 0 and len(low_score) > 0:
        print(f"  HIGH SCORE STOCKS (≥{median_score:.1f}):")
        print(f"    Count:           {len(high_score)}")
        print(f"    Avg Win Rate:    {high_score['win_rate'].mean():.1%}")
        print(f"    Avg 10D Hit:     {high_score['hit_rate_10d'].mean():.1%}")
        print(f"    Avg PF:          {high_score['profit_factor'].mean():.2f}")
        
        print(f"\n  LOW SCORE STOCKS (<{median_score:.1f}):")
        print(f"    Count:           {len(low_score)}")
        print(f"    Avg Win Rate:    {low_score['win_rate'].mean():.1%}")
        print(f"    Avg 10D Hit:     {low_score['hit_rate_10d'].mean():.1%}")
        print(f"    Avg PF:          {low_score['profit_factor'].mean():.2f}")
        
        # Statistical test
        t_stat, t_pvalue = stats.ttest_ind(
            high_score['hit_rate_10d'].dropna(),
            low_score['hit_rate_10d'].dropna()
        )
        
        print(f"\n  T-TEST (High vs Low Score 10D Hit Rate):")
        print(f"    t-statistic:     {t_stat:.3f}")
        print(f"    p-value:         {t_pvalue:.4f}")
        print(f"    Significant:     {'YES' if t_pvalue < 0.05 else 'NO'} (α=0.05)")

print(f"""
CONCLUSION:
───────────────────────────────────────────────────────────────────────────────
""")

if len(valid_analysis) >= 5 and 'hit_rate_10d' in correlations:
    corr = correlations['hit_rate_10d']['correlation']
    pval = correlations['hit_rate_10d']['p_value']
    
    if corr > 0.3 and pval < 0.1:
        verdict = "VALIDATED ✓"
        explanation = "Higher screening scores correlate with better performance"
    elif corr > 0:
        verdict = "PARTIALLY VALIDATED"
        explanation = "Positive but weak correlation observed"
    else:
        verdict = "NOT VALIDATED"
        explanation = "No significant correlation found"
    
    print(f"  Hypothesis: {verdict}")
    print(f"  {explanation}")
    print(f"  Correlation (Screen Score ↔ 10D Hit Rate): r = {corr:.3f}")
else:
    print("  Insufficient data for conclusive validation")

print(f"""
RECOMMENDED STOCKS FOR LIVE TRADING:
───────────────────────────────────────────────────────────────────────────────
""")

# Best candidates
best_candidates = analysis_df[
    (analysis_df['total_trades'] >= 10) &
    (analysis_df['hit_rate_10d'] >= 0.55) &
    (analysis_df['screening_score'] >= 65)
].sort_values('hit_rate_10d', ascending=False)

if len(best_candidates) > 0:
    print("  Top candidates (Screen ≥65, 10D Hit ≥55%, Trades ≥10):\n")
    for _, row in best_candidates.head(10).iterrows():
        print(f"    ✓ {row['ticker']}: Screen={row['screening_score']:.1f}, "
              f"10D Hit={row['hit_rate_10d']:.1%}, PF={row['profit_factor']:.2f}")
else:
    print("  No stocks met all criteria in this validation run")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

# Save all results
analysis_df.to_csv('output/validation_analysis_results.csv', index=False)
print("\nResults saved to output/validation_analysis_results.csv")

# Save comprehensive results
all_results = {
    'screening_df': screening_df,
    'analysis_df': analysis_df,
    'correlations': correlations if 'correlations' in dir() else {},
    'timestamp': datetime.now().isoformat()
}

with open('output/validation_full_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
print("Full results saved to output/validation_full_results.pkl")
