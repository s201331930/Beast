#!/usr/bin/env python3
"""
Stock Anomaly Prediction System - Enhanced with Scientific Prerequisites

This enhanced version includes prerequisite screening to ensure the stock
is suitable for the anomaly prediction system before running the full analysis.

Scientific Prerequisites (empirically validated):
1. HIGH MOMENTUM: Strong price trends with positive momentum
2. HIGH BETA: Beta > 1.2 shows better signal response
3. TRENDING BEHAVIOR: Hurst exponent > 0.55 indicates trending
4. ADEQUATE VOLATILITY: 30-150% annualized for profitable trades
5. RETAIL INTEREST: Higher retail presence improves signal quality
6. BULL REGIME: System performs better in bullish conditions

Usage:
    python main_with_screening.py --ticker SYMBOL [--skip-screening]
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np

from models.stock_screener import StockSuitabilityScreener, StockSuitabilityScore


def run_prerequisite_screening(ticker: str, start_date: str = "2021-08-25") -> tuple:
    """
    Run scientific prerequisite screening before full analysis.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        
    Returns:
        Tuple of (StockSuitabilityScore, should_proceed, data)
    """
    print("\n" + "=" * 70)
    print("PHASE 1: SCIENTIFIC PREREQUISITE SCREENING")
    print("=" * 70)
    
    print(f"\nFetching data for {ticker}...")
    
    # Fetch stock data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date)
    
    if len(df) < 100:
        print(f"ERROR: Insufficient data ({len(df)} days)")
        return None, False, None
    
    df.columns = [c.lower() for c in df.columns]
    df.name = ticker
    
    # Fetch market data for beta
    print("Fetching market data for beta calculation...")
    market = yf.Ticker("^GSPC")
    market_df = market.history(start=start_date)
    market_returns = market_df['Close'].pct_change().dropna()
    
    # Run screening
    print(f"\nRunning scientific screening for {ticker}...")
    screener = StockSuitabilityScreener(df, market_df)
    score = screener.analyze(market_returns)
    
    # Display results
    print(score)
    
    # Determine if we should proceed
    should_proceed = score.overall_score >= 50
    
    # Display recommendation
    print("\n" + "-" * 70)
    print("SCREENING DECISION")
    print("-" * 70)
    
    if score.overall_score >= 75:
        decision = "EXCELLENT CANDIDATE"
        proceed_msg = "Highly recommended to proceed with full analysis"
        icon = "🟢"
    elif score.overall_score >= 60:
        decision = "GOOD CANDIDATE"
        proceed_msg = "Recommended to proceed with full analysis"
        icon = "🟢"
    elif score.overall_score >= 50:
        decision = "MODERATE CANDIDATE"
        proceed_msg = "Proceed with caution - signals may be less reliable"
        icon = "🟡"
    else:
        decision = "POOR CANDIDATE"
        proceed_msg = "Not recommended - consider alternative stocks"
        icon = "🔴"
        should_proceed = False
    
    print(f"\n{icon} {decision}")
    print(f"   Score: {score.overall_score:.1f}/100")
    print(f"   {proceed_msg}")
    
    # Show key factors
    print("\n   Key Factors:")
    factors = [
        ("High Beta", score.is_high_beta, "Enhances signal responsiveness"),
        ("Trending", score.is_trending, "Better for anomaly detection"),
        ("Retail Interest", score.has_retail_interest, "Improves sentiment signals"),
        ("Bull Regime", score.in_bull_regime, "Optimal market conditions")
    ]
    
    for name, status, description in factors:
        status_icon = "✓" if status else "✗"
        print(f"   {status_icon} {name}: {description}")
    
    return score, should_proceed, df


def run_full_analysis(ticker: str, df: pd.DataFrame = None, 
                      skip_backtest: bool = False) -> dict:
    """
    Run the full anomaly prediction analysis.
    
    Args:
        ticker: Stock ticker
        df: Pre-loaded DataFrame (optional)
        skip_backtest: Skip backtesting phase
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 70)
    print("PHASE 2: FULL ANOMALY PREDICTION ANALYSIS")
    print("=" * 70)
    
    from config.settings import config
    from data.collector import DataCollector
    from models.statistical_anomaly import StatisticalAnomalyDetector
    from models.ml_anomaly import MLAnomalyDetector
    from models.cyclical_models import CyclicalAnalyzer
    from models.signal_aggregator import SignalAggregator
    from analysis.sentiment import SentimentAnalyzer
    from analysis.market_context import MarketContextAnalyzer
    from backtest.backtester import Backtester, SignalAnalyzer
    
    # Update config
    config.data.ticker = ticker
    
    # Collect data
    print("\n[1/7] Collecting comprehensive data...")
    collector = DataCollector(ticker)
    data = collector.collect_all_data()
    primary_df = data['primary']
    
    # Statistical Anomaly Detection
    print("\n[2/7] Running statistical anomaly detection...")
    stat_detector = StatisticalAnomalyDetector(primary_df)
    stat_anomalies = stat_detector.run_all_detectors()
    
    # ML Anomaly Detection
    print("\n[3/7] Running ML anomaly detection...")
    ml_detector = MLAnomalyDetector(primary_df)
    ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)
    
    # Cyclical Analysis
    print("\n[4/7] Running cyclical analysis...")
    cyclical_analyzer = CyclicalAnalyzer(primary_df)
    cyclical_signals = cyclical_analyzer.run_all_analysis()
    
    # Sentiment Analysis
    print("\n[5/7] Running sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer(ticker)
    sentiment_signals = sentiment_analyzer.run_full_analysis(primary_df)
    
    # Market Context
    print("\n[6/7] Running market context analysis...")
    market_analyzer = MarketContextAnalyzer(
        primary_df,
        data.get('market', {}),
        data.get('related', {})
    )
    market_signals = market_analyzer.run_full_analysis()
    
    # Signal Aggregation
    print("\n[7/7] Aggregating signals...")
    aggregator = SignalAggregator()
    aggregator.merge_all_signals(
        primary_df,
        stat_anomalies,
        ml_anomalies,
        cyclical_signals,
        sentiment_signals,
        market_signals
    )
    
    trade_signals = aggregator.generate_trade_signals(
        prob_threshold=0.55,
        confidence_threshold=0.3,
        anomaly_threshold=0.4
    )
    analysis_summary = aggregator.get_current_analysis()
    
    results = {
        'primary_df': primary_df,
        'stat_anomalies': stat_anomalies,
        'ml_anomalies': ml_anomalies,
        'cyclical_signals': cyclical_signals,
        'sentiment_signals': sentiment_signals,
        'market_signals': market_signals,
        'trade_signals': trade_signals,
        'analysis_summary': analysis_summary
    }
    
    # Backtesting
    if not skip_backtest:
        print("\n" + "=" * 70)
        print("PHASE 3: BACKTESTING")
        print("=" * 70)
        
        backtester = Backtester()
        backtest_result = backtester.run_backtest(
            primary_df,
            trade_signals,
            signal_column='actionable_signal'
        )
        
        results['backtest_result'] = backtest_result
        
        # Signal analysis
        signal_analyzer = SignalAnalyzer(trade_signals, primary_df['returns'])
        hit_rate = signal_analyzer.analyze_signal_hit_rate('actionable_signal')
        magnitude = signal_analyzer.analyze_signal_magnitude('actionable_signal')
        
        results['hit_rate'] = hit_rate
        results['magnitude'] = magnitude
    
    return results


def print_final_summary(ticker: str, screening_score: StockSuitabilityScore, 
                        results: dict):
    """Print comprehensive final summary."""
    
    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY: {ticker}")
    print("=" * 70)
    
    analysis = results.get('analysis_summary', {})
    backtest = results.get('backtest_result')
    hit_rate = results.get('hit_rate', {})
    
    print(f"""
PREREQUISITE SCREENING:
  Overall Score:       {screening_score.overall_score:.1f}/100 ({screening_score.recommendation})
  Beta:                {screening_score.beta:.2f}
  Hurst Exponent:      {screening_score.hurst_exponent:.3f}
  Key Flags:           {'High Beta' if screening_score.is_high_beta else 'Low Beta'}, 
                       {'Trending' if screening_score.is_trending else 'Non-Trending'}

CURRENT SIGNAL ANALYSIS:
  Rally Probability:   {analysis.get('rally_probability', 0):.1%}
  Anomaly Intensity:   {analysis.get('anomaly_intensity', 0):.1%}
  Directional Bias:    {analysis.get('directional_bias', 0):+.2f}
  Interpretation:      {analysis.get('interpretation', 'N/A')}
""")
    
    if backtest:
        print(f"""BACKTEST PERFORMANCE:
  Total Trades:        {backtest.total_trades}
  Win Rate:            {backtest.win_rate:.1%}
  Profit Factor:       {backtest.profit_factor:.2f}
  Total Return:        {backtest.total_return:.1%}
  Sharpe Ratio:        {backtest.sharpe_ratio:.2f}
  Max Drawdown:        {backtest.max_drawdown:.1%}
""")
    
    if hit_rate:
        print("SIGNAL HIT RATES:")
        for period, rate in hit_rate.items():
            if 'hit_rate' in period:
                print(f"  {period}: {rate:.1%}")
    
    # Final recommendation
    print("\n" + "-" * 70)
    print("RECOMMENDATION")
    print("-" * 70)
    
    if screening_score.overall_score >= 60:
        if backtest and backtest.win_rate > 0.55:
            print(f"✓ {ticker} is a STRONG candidate for the anomaly prediction system")
            print(f"  - High screening score ({screening_score.overall_score:.1f})")
            print(f"  - Positive backtest results ({backtest.win_rate:.1%} win rate)")
        else:
            print(f"✓ {ticker} shows potential but requires monitoring")
    else:
        print(f"✗ {ticker} may not be optimal for this system")
        print(f"  Consider stocks with higher beta and stronger trends")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Stock Anomaly Prediction System with Scientific Prerequisites',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--skip-screening', action='store_true',
                       help='Skip prerequisite screening')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip backtesting phase')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force analysis even if screening fails')
    parser.add_argument('--start-date', type=str, default='2021-08-25',
                       help='Start date for analysis')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STOCK ANOMALY PREDICTION SYSTEM")
    print("Enhanced with Scientific Prerequisites")
    print(f"Analyzing: {args.ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Phase 1: Prerequisite Screening
    if not args.skip_screening:
        screening_score, should_proceed, df = run_prerequisite_screening(
            args.ticker, args.start_date
        )
        
        if not should_proceed and not args.force:
            print("\n" + "=" * 70)
            print("ANALYSIS ABORTED")
            print("=" * 70)
            print(f"\n{args.ticker} does not meet prerequisites for this system.")
            print("Use --force to override this decision.")
            return
        
        if not should_proceed and args.force:
            print("\n⚠️  Proceeding despite low screening score (--force flag)")
    else:
        screening_score = None
        print("\n⚠️  Skipping prerequisite screening (--skip-screening flag)")
    
    # Phase 2: Full Analysis
    results = run_full_analysis(args.ticker, skip_backtest=args.skip_backtest)
    
    # Final Summary
    if screening_score:
        print_final_summary(args.ticker, screening_score, results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
