#!/usr/bin/env python3
"""
Run Full Backtest for RKLB
Comprehensive analysis and backtesting script
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("STOCK ANOMALY PREDICTION SYSTEM - FULL BACKTEST")
print("Analyzing: RKLB (Rocket Lab USA)")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Import modules
from config.settings import config
from data.collector import DataCollector

print("\n[1/8] COLLECTING DATA")
print("-" * 50)

collector = DataCollector("RKLB")
try:
    data = collector.collect_all_data()
    primary_df = data['primary']
    print(f"✓ Collected {len(primary_df)} days of data")
    print(f"✓ Date range: {primary_df.index[0].date()} to {primary_df.index[-1].date()}")
    print(f"✓ Features: {len(primary_df.columns)} columns")
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
# Skip deep learning for faster execution
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

sentiment_analyzer = SentimentAnalyzer("RKLB")
sentiment_signals = sentiment_analyzer.run_full_analysis(primary_df)
sent_summary = sentiment_analyzer.get_sentiment_summary()

print(f"✓ Current sentiment: {sent_summary.get('interpretation', 'N/A')}")
print(f"✓ Composite sentiment: {sent_summary.get('composite_sentiment', 0):.3f}")

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

# Use less conservative thresholds to generate signals
trade_signals = aggregator.generate_trade_signals(
    prob_threshold=0.55,       # Lower from 0.7
    confidence_threshold=0.3,  # Lower from 0.6
    anomaly_threshold=0.4      # Lower from 0.6
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
        save_path='output/report.txt'
    )
    
    # Try to generate charts (may fail without display)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        dashboard.save_all_charts(
            primary_df,
            trade_signals,
            stat_anomalies,
            cyclical_signals,
            backtest_result,
            output_dir='output'
        )
    except Exception as e:
        print(f"Chart generation skipped: {e}")
        
except Exception as e:
    print(f"Report generation error: {e}")

# Final Summary
print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
Ticker: RKLB (Rocket Lab USA)
Analysis Period: {primary_df.index[0].date()} to {primary_df.index[-1].date()}
Trading Days: {len(primary_df)}

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
  Sortino Ratio:       {backtest_result.sortino_ratio:.2f}
  Max Drawdown:        {backtest_result.max_drawdown:.1%}
  Calmar Ratio:        {backtest_result.calmar_ratio:.2f}
  Avg Holding Period:  {backtest_result.avg_holding_period:.1f} days

RISK METRICS:
  Annual Volatility:   {backtest_result.volatility:.1%}
  VaR (95%):           {backtest_result.var_95:.2%}
  CVaR (95%):          {backtest_result.cvar_95:.2%}
""")

if mc_results:
    print(f"""MONTE CARLO ANALYSIS:
  Actual Return Percentile: {mc_results.get('actual_return_percentile', 0):.1f}%
  Actual Sharpe Percentile: {mc_results.get('actual_sharpe_percentile', 0):.1f}%
""")

print("=" * 70)
print("Analysis Complete! Reports saved to output/")
print("=" * 70)

# Save results to pickle for further analysis
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
    'mc_results': mc_results
}

with open('output/results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to output/results.pkl")
