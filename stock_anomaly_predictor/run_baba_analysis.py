#!/usr/bin/env python3
"""
Run Full Analysis for BABA (Alibaba)
Test system predictability on a different stock
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("STOCK ANOMALY PREDICTION SYSTEM - BABA ANALYSIS")
print("Testing system predictability on Alibaba")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Update config for BABA
from config.settings import config
config.data.ticker = "BABA"
config.data.start_date = "2021-08-25"  # Same timeframe as RKLB

# Update related stocks for Chinese tech/e-commerce sector
config.data.related_tickers = [
    "JD",        # JD.com
    "PDD",       # Pinduoduo
    "BIDU",      # Baidu
    "NTES",      # NetEase
    "TCOM",      # Trip.com
    "BILI",      # Bilibili
    "NIO",       # NIO
    "XPEV",      # XPeng
    "LI",        # Li Auto
    "TME",       # Tencent Music
    "AMZN",      # Amazon (competitor)
    "MELI",      # MercadoLibre
]

from data.collector import DataCollector

print("\n[1/8] COLLECTING DATA")
print("-" * 50)

collector = DataCollector("BABA")
try:
    data = collector.collect_all_data()
    primary_df = data['primary']
    print(f"✓ Collected {len(primary_df)} days of data")
    print(f"✓ Date range: {primary_df.index[0].date()} to {primary_df.index[-1].date()}")
    print(f"✓ Features: {len(primary_df.columns)} columns")
    print(f"✓ Current Price: ${primary_df['close'].iloc[-1]:.2f}")
    print(f"✓ 52-week High: ${primary_df['close'].tail(252).max():.2f}")
    print(f"✓ 52-week Low: ${primary_df['close'].tail(252).min():.2f}")
except Exception as e:
    print(f"Error collecting data: {e}")
    sys.exit(1)

# Statistical Anomaly Detection
print("\n[2/8] STATISTICAL ANOMALY DETECTION")
print("-" * 50)

from models.statistical_anomaly import StatisticalAnomalyDetector

stat_detector = StatisticalAnomalyDetector(primary_df)
stat_anomalies = stat_detector.run_all_detectors()
stat_summary = stat_detector.get_anomaly_summary()

print(f"✓ Methods used: {stat_summary['methods_used']}")
print(f"✓ Average anomaly ratio: {stat_summary['avg_anomaly_ratio']:.2%}")
print(f"✓ Max concurrent anomalies: {stat_summary['max_concurrent_anomalies']}")

# ML Anomaly Detection
print("\n[3/8] MACHINE LEARNING ANOMALY DETECTION")
print("-" * 50)

from models.ml_anomaly import MLAnomalyDetector

ml_detector = MLAnomalyDetector(primary_df)
ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)

print(f"✓ ML anomaly detection complete")
print(f"✓ Isolation Forest anomalies: {ml_anomalies['isolation_forest_anomaly'].sum()}")
print(f"✓ LOF anomalies: {ml_anomalies['lof_anomaly'].sum()}")

# Cyclical Analysis
print("\n[4/8] CYCLICAL AND MEAN REVERSION ANALYSIS")
print("-" * 50)

from models.cyclical_models import CyclicalAnalyzer

cyclical_analyzer = CyclicalAnalyzer(primary_df)
cyclical_signals = cyclical_analyzer.run_all_analysis()
cycle_summary = cyclical_analyzer.get_cycle_summary()

hurst = cycle_summary.get('hurst_exponent', 0.5)
if hasattr(hurst, 'iloc'):
    hurst = float(hurst.iloc[-1]) if len(hurst) > 0 else 0.5
print(f"✓ Hurst Exponent: {hurst:.3f}")
if hurst < 0.5:
    print("  → Mean reverting behavior detected")
elif hurst > 0.5:
    print("  → Trending behavior detected")

# Sentiment Analysis
print("\n[5/8] SENTIMENT ANALYSIS")
print("-" * 50)

from analysis.sentiment import SentimentAnalyzer

sentiment_analyzer = SentimentAnalyzer("BABA")
sentiment_signals = sentiment_analyzer.run_full_analysis(primary_df)
sent_summary = sentiment_analyzer.get_sentiment_summary()

print(f"✓ Current sentiment: {sent_summary.get('interpretation', 'N/A')}")

# Market Context Analysis
print("\n[6/8] MARKET CONTEXT ANALYSIS")
print("-" * 50)

from analysis.market_context import MarketContextAnalyzer

market_analyzer = MarketContextAnalyzer(
    primary_df,
    data.get('market', {}),
    data.get('related', {})
)
market_signals = market_analyzer.run_full_analysis()
market_summary = market_analyzer.get_market_summary()

print(f"✓ VIX: {market_summary.get('vix', 'N/A'):.1f}")
print(f"✓ Market Regime: {market_summary.get('market_regime', 'N/A')}")
print(f"✓ Beta: {market_summary.get('beta', 'N/A'):.2f}")

# Signal Aggregation
print("\n[7/8] SIGNAL AGGREGATION")
print("-" * 50)

from models.signal_aggregator import SignalAggregator

aggregator = SignalAggregator()
aggregator.merge_all_signals(
    primary_df,
    stat_anomalies,
    ml_anomalies,
    cyclical_signals,
    sentiment_signals,
    market_signals
)

# Use same thresholds as RKLB analysis
trade_signals = aggregator.generate_trade_signals(
    prob_threshold=0.55,
    confidence_threshold=0.3,
    anomaly_threshold=0.4
)
analysis_summary = aggregator.get_current_analysis()

print(f"✓ Total buy signals: {trade_signals['buy_signal'].sum()}")
print(f"✓ Actionable signals: {trade_signals['actionable_signal'].sum()}")

# Backtesting
print("\n[8/8] COMPREHENSIVE BACKTESTING")
print("-" * 50)

from backtest.backtester import Backtester, SignalAnalyzer

backtester = Backtester()
backtest_result = backtester.run_backtest(
    primary_df,
    trade_signals,
    signal_column='actionable_signal'
)

# Walk-Forward Analysis
print("\nRunning Walk-Forward Validation...")
wf_results = backtester.walk_forward_optimization(
    primary_df,
    trade_signals,
    n_splits=5
)

# Monte Carlo Simulation
if backtest_result.total_trades >= 10:
    mc_results = backtester.monte_carlo_simulation(backtest_result, n_simulations=500)
else:
    mc_results = {}
    print("Insufficient trades for Monte Carlo simulation")

# Benchmark Comparison
if 'returns' in primary_df.columns:
    benchmark_results = backtester.benchmark_comparison(
        backtest_result,
        primary_df['returns']
    )

# Signal Quality Analysis
print("\nAnalyzing Signal Quality...")
signal_analyzer = SignalAnalyzer(trade_signals, primary_df['returns'])

hit_rate = signal_analyzer.analyze_signal_hit_rate('actionable_signal')
magnitude = signal_analyzer.analyze_signal_magnitude('actionable_signal', forward_days=5)

print("\nSignal Hit Rates:")
for period, rate in hit_rate.items():
    if 'hit_rate' in period:
        print(f"  {period}: {rate:.1%}")

print("\nSignal Magnitude Analysis:")
print(f"  Avg signal return: {magnitude.get('avg_signal_return', 0):.2%}")
print(f"  Avg non-signal return: {magnitude.get('avg_non_signal_return', 0):.2%}")
print(f"  Lift: {magnitude.get('lift', 0):.2f} std devs")
if 'statistically_significant' in magnitude:
    print(f"  Statistically significant: {magnitude['statistically_significant']}")

# ============================================================
# HIGHLIGHT SIGNAL PERIODS
# ============================================================
print("\n" + "=" * 70)
print("SIGNAL PERIODS ANALYSIS - BABA")
print("=" * 70)

# Get all actionable signals
signal_dates = trade_signals[trade_signals['actionable_signal'] == True].index
print(f"\nTotal Actionable Signals: {len(signal_dates)}")

if len(signal_dates) > 0:
    print("\n" + "-" * 70)
    print("DETAILED SIGNAL BREAKDOWN BY DATE")
    print("-" * 70)
    
    signal_details = []
    
    for i, date in enumerate(signal_dates):
        idx = primary_df.index.get_loc(date)
        
        # Get price at signal
        price_at_signal = primary_df['close'].iloc[idx]
        
        # Calculate forward returns
        fwd_1d = fwd_5d = fwd_10d = fwd_20d = np.nan
        
        if idx + 1 < len(primary_df):
            fwd_1d = (primary_df['close'].iloc[idx+1] / price_at_signal - 1) * 100
        if idx + 5 < len(primary_df):
            fwd_5d = (primary_df['close'].iloc[idx+5] / price_at_signal - 1) * 100
        if idx + 10 < len(primary_df):
            fwd_10d = (primary_df['close'].iloc[idx+10] / price_at_signal - 1) * 100
        if idx + 20 < len(primary_df):
            fwd_20d = (primary_df['close'].iloc[idx+20] / price_at_signal - 1) * 100
        
        # Get signal strength
        strength = trade_signals['signal_strength'].iloc[idx]
        composite = trade_signals['composite_score'].iloc[idx]
        
        signal_details.append({
            'date': date,
            'price': price_at_signal,
            'composite_score': composite,
            'signal_strength': strength,
            'fwd_1d': fwd_1d,
            'fwd_5d': fwd_5d,
            'fwd_10d': fwd_10d,
            'fwd_20d': fwd_20d,
            'win_5d': 1 if fwd_5d > 0 else 0,
            'win_10d': 1 if fwd_10d > 0 else 0
        })
    
    signals_df = pd.DataFrame(signal_details)
    
    # Print each signal
    print(f"\n{'Date':<12} {'Price':>8} {'Score':>7} {'1D':>8} {'5D':>8} {'10D':>8} {'20D':>8} {'Result'}")
    print("-" * 80)
    
    for _, row in signals_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        result = "✓ WIN" if row['fwd_10d'] > 0 else "✗ LOSS" if not np.isnan(row['fwd_10d']) else "PENDING"
        print(f"{date_str:<12} ${row['price']:>7.2f} {row['composite_score']:>6.1f} "
              f"{row['fwd_1d']:>+7.1f}% {row['fwd_5d']:>+7.1f}% "
              f"{row['fwd_10d']:>+7.1f}% {row['fwd_20d']:>+7.1f}% {result}")
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("SIGNAL PERFORMANCE SUMMARY")
    print("-" * 70)
    
    valid_5d = signals_df[~signals_df['fwd_5d'].isna()]
    valid_10d = signals_df[~signals_df['fwd_10d'].isna()]
    valid_20d = signals_df[~signals_df['fwd_20d'].isna()]
    
    print(f"\n5-Day Performance (n={len(valid_5d)}):")
    if len(valid_5d) > 0:
        print(f"  Win Rate: {valid_5d['win_5d'].mean():.1%}")
        print(f"  Avg Return: {valid_5d['fwd_5d'].mean():+.2f}%")
        print(f"  Best: {valid_5d['fwd_5d'].max():+.2f}%")
        print(f"  Worst: {valid_5d['fwd_5d'].min():+.2f}%")
    
    print(f"\n10-Day Performance (n={len(valid_10d)}):")
    if len(valid_10d) > 0:
        print(f"  Win Rate: {valid_10d['win_10d'].mean():.1%}")
        print(f"  Avg Return: {valid_10d['fwd_10d'].mean():+.2f}%")
        print(f"  Best: {valid_10d['fwd_10d'].max():+.2f}%")
        print(f"  Worst: {valid_10d['fwd_10d'].min():+.2f}%")
    
    print(f"\n20-Day Performance (n={len(valid_20d)}):")
    if len(valid_20d) > 0:
        print(f"  Avg Return: {valid_20d['fwd_20d'].mean():+.2f}%")
        print(f"  Best: {valid_20d['fwd_20d'].max():+.2f}%")
        print(f"  Worst: {valid_20d['fwd_20d'].min():+.2f}%")
    
    # Identify best and worst signal periods
    print("\n" + "-" * 70)
    print("NOTABLE SIGNAL PERIODS")
    print("-" * 70)
    
    if len(valid_10d) > 0:
        best_signal = valid_10d.loc[valid_10d['fwd_10d'].idxmax()]
        worst_signal = valid_10d.loc[valid_10d['fwd_10d'].idxmin()]
        
        print(f"\n📈 BEST SIGNAL:")
        print(f"   Date: {best_signal['date'].strftime('%Y-%m-%d')}")
        print(f"   Price: ${best_signal['price']:.2f}")
        print(f"   10-Day Return: {best_signal['fwd_10d']:+.2f}%")
        
        print(f"\n📉 WORST SIGNAL:")
        print(f"   Date: {worst_signal['date'].strftime('%Y-%m-%d')}")
        print(f"   Price: ${worst_signal['price']:.2f}")
        print(f"   10-Day Return: {worst_signal['fwd_10d']:+.2f}%")
    
    # Group signals by year
    print("\n" + "-" * 70)
    print("SIGNALS BY YEAR")
    print("-" * 70)
    
    signals_df['year'] = signals_df['date'].dt.year
    yearly_stats = signals_df.groupby('year').agg({
        'date': 'count',
        'fwd_5d': 'mean',
        'fwd_10d': 'mean',
        'win_10d': 'mean'
    }).rename(columns={'date': 'count'})
    
    print(f"\n{'Year':<6} {'Signals':>8} {'Avg 5D':>10} {'Avg 10D':>10} {'Win Rate':>10}")
    print("-" * 50)
    for year, row in yearly_stats.iterrows():
        print(f"{year:<6} {row['count']:>8.0f} {row['fwd_5d']:>+9.2f}% {row['fwd_10d']:>+9.2f}% {row['win_10d']:>9.1%}")
    
    # Save signal details
    signals_df.to_csv('output/baba_signals.csv', index=False)
    print(f"\n✓ Signal details saved to output/baba_signals.csv")

else:
    print("\nNo actionable signals generated for BABA in this period.")

# Generate Reports
print("\n" + "=" * 70)
print("GENERATING REPORTS")
print("=" * 70)

os.makedirs('output', exist_ok=True)

from visualization.dashboard import DashboardGenerator

dashboard = DashboardGenerator()

try:
    # Generate text report
    report = dashboard.generate_summary_report(
        backtest_result,
        analysis_summary,
        save_path='output/baba_report.txt'
    )
    
    # Generate charts
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        dashboard.plot_price_with_signals(
            primary_df, trade_signals,
            save_path='output/baba_price_signals.png'
        )
        
        dashboard.plot_anomaly_analysis(
            primary_df, stat_anomalies,
            save_path='output/baba_anomaly_analysis.png'
        )
        
    except Exception as e:
        print(f"Chart generation note: {e}")
        
except Exception as e:
    print(f"Report generation error: {e}")

# Final Summary
print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY - BABA")
print("=" * 70)

print(f"""
Ticker: BABA (Alibaba Group)
Analysis Period: {primary_df.index[0].date()} to {primary_df.index[-1].date()}
Trading Days: {len(primary_df)}
Current Price: ${primary_df['close'].iloc[-1]:.2f}

CURRENT SIGNAL ANALYSIS:
  Composite Score:     {analysis_summary.get('rally_probability', 0) * 100:.1f}/100
  Rally Probability:   {analysis_summary.get('rally_probability', 0):.1%}
  Anomaly Intensity:   {analysis_summary.get('anomaly_intensity', 0):.1%}
  Signal Confidence:   {analysis_summary.get('signal_confidence', 0):.1%}
  Directional Bias:    {analysis_summary.get('directional_bias', 0):+.2f}
  Interpretation:      {analysis_summary.get('interpretation', 'N/A')}

BACKTEST PERFORMANCE:
  Total Trades:        {backtest_result.total_trades}
  Win Rate:            {backtest_result.win_rate:.1%}
  Profit Factor:       {backtest_result.profit_factor:.2f}
  Total Return:        {backtest_result.total_return:.1%}
  Annual Return:       {backtest_result.annual_return:.1%}
  Sharpe Ratio:        {backtest_result.sharpe_ratio:.2f}
  Max Drawdown:        {backtest_result.max_drawdown:.1%}
  Avg Holding Period:  {backtest_result.avg_holding_period:.1f} days

CYCLICAL CHARACTERISTICS:
  Hurst Exponent:      {hurst:.3f} ({'Trending' if hurst > 0.5 else 'Mean Reverting'})
  Beta vs S&P 500:     {market_summary.get('beta', 0):.2f}
""")

if mc_results:
    print(f"""MONTE CARLO ANALYSIS:
  Actual Return Percentile: {mc_results.get('actual_return_percentile', 0):.1f}%
  Actual Sharpe Percentile: {mc_results.get('actual_sharpe_percentile', 0):.1f}%
""")

# COMPARISON WITH RKLB
print("=" * 70)
print("COMPARISON: BABA vs RKLB")
print("=" * 70)
print("""
Metric                    BABA            RKLB
---------------------------------------------------------""")
print(f"Hurst Exponent            {hurst:.3f}           1.040")
print(f"Beta                      {market_summary.get('beta', 0):.2f}            4.52")
print(f"Total Signals             {len(signal_dates)}              46")
print(f"Backtest Win Rate         {backtest_result.win_rate:.1%}           47.8%")
print(f"Profit Factor             {backtest_result.profit_factor:.2f}            1.96")

print("\n" + "=" * 70)
print("Analysis Complete! Reports saved to output/")
print("=" * 70)

# Save all results
import pickle
results = {
    'primary_df': primary_df,
    'stat_anomalies': stat_anomalies,
    'ml_anomalies': ml_anomalies,
    'cyclical_signals': cyclical_signals,
    'sentiment_signals': sentiment_signals,
    'market_signals': market_signals,
    'trade_signals': trade_signals,
    'backtest_result': backtest_result,
    'analysis_summary': analysis_summary,
    'wf_results': wf_results,
    'mc_results': mc_results,
    'signal_dates': signal_dates
}

with open('output/baba_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to output/baba_results.pkl")
