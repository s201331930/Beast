#!/usr/bin/env python3
"""
Stock Anomaly Prediction System - Main Pipeline

State-of-the-art anomaly detection and rally prediction system
combining mathematical, statistical, and physics-inspired models.

This system analyzes:
- Price and volume anomalies
- Cyclical patterns and mean reversion
- Sentiment from multiple sources
- Market context and correlations
- Machine learning anomaly detection

To predict potential big stock moves (rallies) before they happen.

Usage:
    python main.py [--ticker TICKER] [--backtest] [--full-analysis]
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import config, Config
from data.collector import DataCollector
from models.statistical_anomaly import StatisticalAnomalyDetector
from models.ml_anomaly import MLAnomalyDetector
from models.cyclical_models import CyclicalAnalyzer
from models.signal_aggregator import SignalAggregator
from analysis.sentiment import SentimentAnalyzer
from analysis.market_context import MarketContextAnalyzer
from backtest.backtester import Backtester, SignalAnalyzer
from visualization.dashboard import DashboardGenerator


class StockAnomalyPredictor:
    """
    Main orchestrator for the Stock Anomaly Prediction System.
    """
    
    def __init__(self, ticker: str = None):
        """
        Initialize the prediction system.
        
        Args:
            ticker: Stock ticker symbol (default: from config)
        """
        self.ticker = ticker or config.data.ticker
        self.data = {}
        self.results = {}
        
        print("=" * 70)
        print(f"  STOCK ANOMALY PREDICTION SYSTEM")
        print(f"  Analyzing: {self.ticker}")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def collect_data(self) -> dict:
        """
        Collect all data from various sources.
        
        Returns:
            Dictionary with all collected data
        """
        print("\n[1/7] COLLECTING DATA")
        print("-" * 50)
        
        collector = DataCollector(self.ticker)
        self.data = collector.collect_all_data()
        
        return self.data
    
    def run_statistical_analysis(self) -> pd.DataFrame:
        """
        Run statistical anomaly detection.
        
        Returns:
            DataFrame with statistical anomaly scores
        """
        print("\n[2/7] STATISTICAL ANOMALY DETECTION")
        print("-" * 50)
        
        detector = StatisticalAnomalyDetector(self.data['primary'])
        self.results['statistical_anomalies'] = detector.run_all_detectors()
        
        summary = detector.get_anomaly_summary()
        print(f"\nSummary: {summary['methods_used']} methods, "
              f"avg anomaly ratio: {summary['avg_anomaly_ratio']:.2%}")
        
        return self.results['statistical_anomalies']
    
    def run_ml_analysis(self, include_deep_learning: bool = True) -> pd.DataFrame:
        """
        Run machine learning anomaly detection.
        
        Args:
            include_deep_learning: Whether to include autoencoder methods
            
        Returns:
            DataFrame with ML anomaly scores
        """
        print("\n[3/7] MACHINE LEARNING ANOMALY DETECTION")
        print("-" * 50)
        
        detector = MLAnomalyDetector(self.data['primary'])
        self.results['ml_anomalies'] = detector.run_all_detectors(
            include_deep_learning=include_deep_learning
        )
        
        return self.results['ml_anomalies']
    
    def run_cyclical_analysis(self) -> pd.DataFrame:
        """
        Run cyclical and mean reversion analysis.
        
        Returns:
            DataFrame with cyclical signals
        """
        print("\n[4/7] CYCLICAL AND MEAN REVERSION ANALYSIS")
        print("-" * 50)
        
        analyzer = CyclicalAnalyzer(self.data['primary'])
        self.results['cyclical_signals'] = analyzer.run_all_analysis()
        
        summary = analyzer.get_cycle_summary()
        print(f"\nHurst Exponent: {summary.get('hurst_exponent', 'N/A')}")
        
        return self.results['cyclical_signals']
    
    def run_sentiment_analysis(self) -> pd.DataFrame:
        """
        Run sentiment analysis from multiple sources.
        
        Returns:
            DataFrame with sentiment signals
        """
        print("\n[5/7] SENTIMENT ANALYSIS")
        print("-" * 50)
        
        analyzer = SentimentAnalyzer(self.ticker)
        self.results['sentiment_signals'] = analyzer.run_full_analysis(
            self.data['primary']
        )
        
        return self.results['sentiment_signals']
    
    def run_market_analysis(self) -> pd.DataFrame:
        """
        Run market context analysis.
        
        Returns:
            DataFrame with market context signals
        """
        print("\n[6/7] MARKET CONTEXT ANALYSIS")
        print("-" * 50)
        
        analyzer = MarketContextAnalyzer(
            self.data['primary'],
            self.data.get('market', {}),
            self.data.get('related', {})
        )
        self.results['market_signals'] = analyzer.run_full_analysis()
        
        summary = analyzer.get_market_summary()
        print(f"\nVIX: {summary.get('vix', 'N/A'):.1f}, "
              f"Regime: {summary.get('market_regime', 'N/A')}")
        
        return self.results['market_signals']
    
    def aggregate_signals(self) -> pd.DataFrame:
        """
        Aggregate all signals into final predictions.
        
        Returns:
            DataFrame with aggregated signals and trade recommendations
        """
        print("\n[7/7] SIGNAL AGGREGATION")
        print("-" * 50)
        
        aggregator = SignalAggregator()
        
        # Merge all signals
        aggregator.merge_all_signals(
            self.data['primary'],
            self.results.get('statistical_anomalies', pd.DataFrame()),
            self.results.get('ml_anomalies', pd.DataFrame()),
            self.results.get('cyclical_signals', pd.DataFrame()),
            self.results.get('sentiment_signals', pd.DataFrame()),
            self.results.get('market_signals', pd.DataFrame())
        )
        
        # Generate trade signals
        self.results['trade_signals'] = aggregator.generate_trade_signals()
        self.results['analysis_summary'] = aggregator.get_current_analysis()
        
        return self.results['trade_signals']
    
    def run_backtest(self) -> dict:
        """
        Run comprehensive backtest on generated signals.
        
        Returns:
            Dictionary with backtest results
        """
        print("\n" + "=" * 70)
        print("BACKTESTING")
        print("=" * 70)
        
        backtester = Backtester()
        
        # Run main backtest
        backtest_result = backtester.run_backtest(
            self.data['primary'],
            self.results['trade_signals'],
            signal_column='actionable_signal'
        )
        
        self.results['backtest'] = backtest_result
        
        # Walk-forward analysis
        print("\nRunning walk-forward validation...")
        wf_results = backtester.walk_forward_optimization(
            self.data['primary'],
            self.results['trade_signals'],
            n_splits=5
        )
        self.results['walk_forward'] = wf_results
        
        # Monte Carlo simulation
        mc_results = backtester.monte_carlo_simulation(backtest_result)
        self.results['monte_carlo'] = mc_results
        
        # Benchmark comparison
        if 'returns' in self.data['primary'].columns:
            benchmark_comp = backtester.benchmark_comparison(
                backtest_result,
                self.data['primary']['returns']
            )
            self.results['benchmark_comparison'] = benchmark_comp
        
        # Signal analysis
        signal_analyzer = SignalAnalyzer(
            self.results['trade_signals'],
            self.data['primary']['returns'] if 'returns' in self.data['primary'].columns else self.data['primary']['close'].pct_change()
        )
        
        hit_rate = signal_analyzer.analyze_signal_hit_rate('actionable_signal')
        magnitude = signal_analyzer.analyze_signal_magnitude('actionable_signal')
        
        self.results['signal_analysis'] = {
            'hit_rate': hit_rate,
            'magnitude': magnitude
        }
        
        return self.results
    
    def generate_reports(self, output_dir: str = 'output'):
        """
        Generate visualization reports.
        
        Args:
            output_dir: Output directory for reports
        """
        print("\n" + "=" * 70)
        print("GENERATING REPORTS")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        dashboard = DashboardGenerator()
        
        # Generate charts
        dashboard.save_all_charts(
            self.data['primary'],
            self.results.get('trade_signals', pd.DataFrame()),
            self.results.get('statistical_anomalies', pd.DataFrame()),
            self.results.get('cyclical_signals', pd.DataFrame()),
            self.results.get('backtest'),
            output_dir
        )
        
        # Generate text report
        report = dashboard.generate_summary_report(
            self.results.get('backtest'),
            self.results.get('analysis_summary', {}),
            save_path=f"{output_dir}/report.txt"
        )
        
        print(f"\nReports saved to: {output_dir}/")
    
    def get_current_signal(self) -> dict:
        """
        Get the current signal and recommendation.
        
        Returns:
            Dictionary with current signal analysis
        """
        if 'analysis_summary' not in self.results:
            return {}
        
        analysis = self.results['analysis_summary']
        trade_signals = self.results.get('trade_signals', pd.DataFrame())
        
        current = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': self.ticker,
            **analysis
        }
        
        if not trade_signals.empty:
            latest = trade_signals.iloc[-1]
            current['composite_score'] = latest.get('composite_score', 0)
            current['alert_level'] = latest.get('alert_level', 'unknown')
            current['actionable_signal'] = latest.get('actionable_signal', False)
        
        return current
    
    def run_full_pipeline(self,
                          include_deep_learning: bool = True,
                          run_backtest: bool = True,
                          generate_reports: bool = True,
                          output_dir: str = 'output') -> dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            include_deep_learning: Include autoencoder methods
            run_backtest: Whether to run backtesting
            generate_reports: Whether to generate reports
            output_dir: Output directory
            
        Returns:
            Dictionary with all results
        """
        # Collect data
        self.collect_data()
        
        # Run all analysis modules
        self.run_statistical_analysis()
        self.run_ml_analysis(include_deep_learning)
        self.run_cyclical_analysis()
        self.run_sentiment_analysis()
        self.run_market_analysis()
        
        # Aggregate signals
        self.aggregate_signals()
        
        # Backtest
        if run_backtest:
            self.run_backtest()
        
        # Generate reports
        if generate_reports:
            self.generate_reports(output_dir)
        
        # Print final summary
        self._print_final_summary()
        
        return self.results
    
    def _print_final_summary(self):
        """Print final analysis summary."""
        print("\n" + "=" * 70)
        print("FINAL ANALYSIS SUMMARY")
        print("=" * 70)
        
        current = self.get_current_signal()
        
        print(f"\nTicker: {self.ticker}")
        print(f"Date: {current.get('date', 'N/A')}")
        print(f"\nCurrent Signal Analysis:")
        print(f"  Composite Score:     {current.get('composite_score', 0):.1f}/100")
        print(f"  Alert Level:         {current.get('alert_level', 'N/A')}")
        print(f"  Rally Probability:   {current.get('rally_probability', 0):.1%}")
        print(f"  Anomaly Intensity:   {current.get('anomaly_intensity', 0):.1%}")
        print(f"  Signal Confidence:   {current.get('signal_confidence', 0):.1%}")
        print(f"  Directional Bias:    {current.get('directional_bias', 0):+.2f}")
        print(f"\nInterpretation: {current.get('interpretation', 'N/A')}")
        
        if current.get('actionable_signal'):
            print(f"\n*** ACTIONABLE BUY SIGNAL DETECTED ***")
        
        # Backtest summary if available
        if 'backtest' in self.results:
            bt = self.results['backtest']
            print(f"\nBacktest Performance:")
            print(f"  Sharpe Ratio:  {bt.sharpe_ratio:.2f}")
            print(f"  Total Return:  {bt.total_return:.1%}")
            print(f"  Win Rate:      {bt.win_rate:.1%}")
            print(f"  Max Drawdown:  {bt.max_drawdown:.1%}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Stock Anomaly Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                      # Analyze default ticker (RKLB)
    python main.py --ticker TSLA        # Analyze TSLA
    python main.py --backtest           # Run with backtesting
    python main.py --full-analysis      # Run complete analysis
    python main.py --quick              # Quick analysis (no deep learning)
        """
    )
    
    parser.add_argument('--ticker', '-t', type=str, default='RKLB',
                       help='Stock ticker symbol (default: RKLB)')
    parser.add_argument('--backtest', '-b', action='store_true',
                       help='Run backtesting')
    parser.add_argument('--full-analysis', '-f', action='store_true',
                       help='Run full analysis with reports')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick analysis (skip deep learning)')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StockAnomalyPredictor(args.ticker)
    
    # Run analysis
    if args.full_analysis:
        predictor.run_full_pipeline(
            include_deep_learning=not args.quick,
            run_backtest=True,
            generate_reports=True,
            output_dir=args.output
        )
    else:
        predictor.run_full_pipeline(
            include_deep_learning=not args.quick,
            run_backtest=args.backtest,
            generate_reports=args.backtest,
            output_dir=args.output
        )
    
    return predictor.results


if __name__ == "__main__":
    results = main()
