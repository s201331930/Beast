#!/usr/bin/env python3
"""
RKLB Stock Anomaly Detection System - Main Runner
=================================================
State-of-the-art anomaly detection system for predicting big stock moves.

This script runs the complete analysis pipeline:
1. Data collection from multiple sources
2. Statistical anomaly detection
3. Machine learning anomaly detection
4. Cyclical/mean reversion analysis
5. Sentiment analysis
6. Ensemble signal generation
7. Backtesting and validation
8. Report generation

Author: Quantitative Research Team
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data.data_collector import create_comprehensive_dataset, MarketDataCollector
from models.statistical_models import StatisticalAnomalyEnsemble
from models.ml_models import MLAnomalyEnsemble
from models.cyclical_models import CyclicalModelEnsemble
from analysis.technical_indicators import TechnicalIndicators
from analysis.sentiment_analyzer import SentimentAnalyzer, VIXAnalyzer
from analysis.ensemble_signal_generator import EnsembleSignalGenerator, RallyPredictor, SignalConfig
from backtest.backtester import SignalBacktester, BacktestConfig, MonteCarloSimulator, EventStudyAnalyzer
from visualization.dashboard import AnomalyVisualizer, create_full_report


def setup_logging(log_dir: str = "logs"):
    """Configure logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.add(log_file, rotation="100 MB", level="INFO")
    logger.info("="*60)
    logger.info("RKLB Anomaly Detection System Started")
    logger.info("="*60)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return {}


def run_full_analysis(ticker: str = "RKLB", 
                      start_date: str = "2021-08-25",
                      output_dir: str = "reports"):
    """
    Run the complete anomaly detection analysis pipeline.
    """
    logger.info(f"Starting full analysis for {ticker}")
    logger.info(f"Analysis period: {start_date} to present")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/charts", exist_ok=True)
    
    # ============================================
    # STEP 1: Data Collection
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Data Collection")
    logger.info("="*50)
    
    try:
        # Collect comprehensive market data
        data = create_comprehensive_dataset(
            ticker=ticker,
            start_date=start_date,
            save_path=f"{output_dir}/data/{ticker}_raw_data.csv"
        )
        
        logger.info(f"Collected {len(data)} days of data")
        logger.info(f"Columns: {len(data.columns)}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
    
    # ============================================
    # STEP 2: Generate Ensemble Signals
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Ensemble Signal Generation")
    logger.info("="*50)
    
    # Configure signal generator
    signal_config = SignalConfig(
        min_stat_models=2,
        min_ml_models=2,
        min_cyclical_models=1,
        min_sentiment_signals=1,
        stat_weight=0.25,
        ml_weight=0.25,
        cyclical_weight=0.25,
        sentiment_weight=0.15,
        technical_weight=0.10,
        signal_threshold=0.45,
        strong_signal_threshold=0.65,
        require_volume_confirmation=True,
        signal_cooldown=3
    )
    
    generator = EnsembleSignalGenerator(config=signal_config)
    
    try:
        all_features, signals = generator.generate_signals(data)
        
        # Get signal summary
        summary = generator.get_signal_summary()
        
        logger.info(f"\n--- Signal Summary ---")
        logger.info(f"Total signals generated: {summary.get('total_signals', 0)}")
        logger.info(f"Strong signals: {summary.get('strong_signals', 0)}")
        logger.info(f"Regular signals: {summary.get('regular_signals', 0)}")
        logger.info(f"Signal rate: {summary.get('signal_rate', 0)*100:.2f}%")
        
        # Save features
        all_features.to_csv(f"{output_dir}/data/{ticker}_all_features.csv")
        signals.to_csv(f"{output_dir}/data/{ticker}_signals.csv")
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # STEP 3: Backtesting
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Backtesting")
    logger.info("="*50)
    
    # Configure backtester
    backtest_config = BacktestConfig(
        initial_capital=100000,
        position_size=0.15,  # 15% per position
        max_positions=3,
        stop_loss=0.08,      # 8% stop loss
        take_profit=0.25,    # 25% take profit
        trailing_stop=0.10,  # 10% trailing stop
        commission=0.001,
        slippage=0.001,
        holding_period=30,
        min_holding_period=2
    )
    
    backtester = SignalBacktester(backtest_config)
    
    try:
        # Run backtest
        results = backtester.backtest(data, signals, signal_name=f"{ticker}_ensemble")
        
        # Print report
        report = backtester.generate_report(results)
        logger.info(f"\n{report}")
        
        # Save report
        with open(f"{output_dir}/backtest_report.txt", 'w') as f:
            f.write(report)
        
        # Save equity curve
        results.equity_curve.to_csv(f"{output_dir}/data/equity_curve.csv")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        results = None
    
    # ============================================
    # STEP 4: Rally Prediction Analysis
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Rally Prediction Analysis")
    logger.info("="*50)
    
    try:
        predictor = RallyPredictor()
        predictions = predictor.predict_rallies(
            data, 
            rally_threshold=0.15,  # 15% rally threshold
            lookforward=20         # 20-day forward window
        )
        
        metrics = predictor.get_prediction_metrics(predictions)
        
        logger.info("\n--- Rally Prediction Metrics ---")
        logger.info(f"Total signals: {metrics.get('total_signals', 0)}")
        logger.info(f"Actual rallies in data: {metrics.get('actual_rallies', 0)}")
        logger.info(f"True positives: {metrics.get('true_positives', 0)}")
        logger.info(f"False positives: {metrics.get('false_positives', 0)}")
        logger.info(f"Precision: {metrics.get('precision', 0)*100:.1f}%")
        logger.info(f"Recall: {metrics.get('recall', 0)*100:.1f}%")
        logger.info(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
        logger.info(f"Avg return when signal fires: {metrics.get('avg_return_on_signal', 0)*100:.1f}%")
        
        # Save predictions
        predictions.to_csv(f"{output_dir}/data/rally_predictions.csv")
        
    except Exception as e:
        logger.error(f"Rally prediction failed: {e}")
        predictions = None
        metrics = {}
    
    # ============================================
    # STEP 5: Monte Carlo Simulation
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 5: Monte Carlo Simulation")
    logger.info("="*50)
    
    try:
        if results and results.trades:
            mc_simulator = MonteCarloSimulator(n_simulations=1000)
            mc_results = mc_simulator.simulate(results.trades)
            
            logger.info("\n--- Monte Carlo Results ---")
            logger.info(f"Mean Return: {mc_results.get('mean_return', 0)*100:.1f}%")
            logger.info(f"Median Return: {mc_results.get('median_return', 0)*100:.1f}%")
            logger.info(f"5th Percentile: {mc_results.get('return_5th_percentile', 0)*100:.1f}%")
            logger.info(f"95th Percentile: {mc_results.get('return_95th_percentile', 0)*100:.1f}%")
            logger.info(f"Probability of Positive Return: {mc_results.get('probability_positive', 0)*100:.1f}%")
            logger.info(f"Mean Max Drawdown: {mc_results.get('mean_max_drawdown', 0)*100:.1f}%")
            logger.info(f"Worst Case Drawdown: {mc_results.get('worst_drawdown', 0)*100:.1f}%")
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
    
    # ============================================
    # STEP 6: Event Study Analysis
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 6: Event Study Analysis")
    logger.info("="*50)
    
    try:
        event_analyzer = EventStudyAnalyzer(pre_window=10, post_window=20)
        event_results = event_analyzer.analyze(data, signals)
        
        if not event_results.empty:
            logger.info("\n--- Event Study Results (Average Returns Around Signals) ---")
            logger.info(f"Pre-signal 5-day return: {event_results.loc[-5, 'cumulative_return']*100:.2f}%")
            logger.info(f"Signal day return: {event_results.loc[0, 'avg_return']*100:.2f}%")
            logger.info(f"Post-signal 5-day return: {event_results.loc[5, 'cumulative_return']*100:.2f}%")
            logger.info(f"Post-signal 10-day return: {event_results.loc[10, 'cumulative_return']*100:.2f}%")
            logger.info(f"Post-signal 20-day return: {event_results.loc[20, 'cumulative_return']*100:.2f}%")
            
            # Save event study
            event_results.to_csv(f"{output_dir}/data/event_study.csv")
    except Exception as e:
        logger.error(f"Event study failed: {e}")
    
    # ============================================
    # STEP 7: Generate Visualizations
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 7: Generating Visualizations")
    logger.info("="*50)
    
    try:
        visualizer = AnomalyVisualizer()
        
        # Price chart with signals
        visualizer.plot_price_with_signals(
            all_features, signals,
            title=f"{ticker} Anomaly Detection Signals",
            save_path=f"{output_dir}/charts/price_signals.png"
        )
        logger.info("Created: price_signals.png")
        
        # Anomaly heatmap
        visualizer.plot_anomaly_heatmap(
            all_features,
            title=f"{ticker} Anomaly Detection Heatmap",
            save_path=f"{output_dir}/charts/anomaly_heatmap.png"
        )
        logger.info("Created: anomaly_heatmap.png")
        
        # Model comparison
        visualizer.plot_model_comparison(
            all_features,
            title=f"{ticker} Model Signal Comparison",
            save_path=f"{output_dir}/charts/model_comparison.png"
        )
        logger.info("Created: model_comparison.png")
        
        # Backtest results
        if results:
            visualizer.plot_backtest_results(
                results.equity_curve,
                data['close'],
                results.drawdown_curve,
                results.trades,
                title=f"{ticker} Strategy Backtest Performance",
                save_path=f"{output_dir}/charts/backtest_results.png"
            )
            logger.info("Created: backtest_results.png")
        
        # Signal analysis
        if predictions is not None:
            visualizer.plot_signal_analysis(
                predictions,
                title=f"{ticker} Signal Quality Analysis",
                save_path=f"{output_dir}/charts/signal_analysis.png"
            )
            logger.info("Created: signal_analysis.png")
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================
    # STEP 8: Generate Final Report
    # ============================================
    logger.info("\n" + "="*50)
    logger.info("STEP 8: Generating Final Report")
    logger.info("="*50)
    
    # Create summary report
    summary_report = f"""
================================================================================
                    RKLB ANOMALY DETECTION ANALYSIS REPORT
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

EXECUTIVE SUMMARY
-----------------
Ticker: {ticker}
Analysis Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
Total Trading Days: {len(data)}

SIGNAL GENERATION
-----------------
Total Features Generated: {len(all_features.columns)}
Total Buy Signals: {summary.get('total_signals', 0)}
Strong Signals (score > 0.65): {summary.get('strong_signals', 0)}
Regular Signals (score 0.45-0.65): {summary.get('regular_signals', 0)}
Signal Frequency: {summary.get('signal_rate', 0)*100:.2f}% of trading days

BACKTEST PERFORMANCE
--------------------
"""
    
    if results:
        summary_report += f"""Total Return: {results.total_return * 100:.2f}%
Annualized Return: {results.annualized_return * 100:.2f}%
Benchmark Return (Buy & Hold): {results.benchmark_return * 100:.2f}%
Excess Return: {results.excess_return * 100:.2f}%

Risk Metrics:
- Sharpe Ratio: {results.sharpe_ratio:.3f}
- Sortino Ratio: {results.sortino_ratio:.3f}
- Max Drawdown: {results.max_drawdown * 100:.2f}%
- Volatility: {results.volatility * 100:.2f}%

Trade Statistics:
- Total Trades: {results.total_trades}
- Win Rate: {results.win_rate * 100:.1f}%
- Profit Factor: {results.profit_factor:.2f}
- Avg Win: {results.avg_win * 100:.2f}%
- Avg Loss: {results.avg_loss * 100:.2f}%
- Avg Holding Period: {results.avg_holding_period:.1f} days
"""

    summary_report += f"""
RALLY PREDICTION METRICS
------------------------
Precision: {metrics.get('precision', 0)*100:.1f}%
Recall: {metrics.get('recall', 0)*100:.1f}%
F1 Score: {metrics.get('f1_score', 0):.3f}
Avg Return on Signal: {metrics.get('avg_return_on_signal', 0)*100:.1f}%

MODELS EMPLOYED
---------------
Statistical Models:
- Z-Score Anomaly Detection
- Modified Z-Score (MAD-based)
- Bollinger Bands
- GARCH Volatility Model
- Mahalanobis Distance
- Extreme Value Theory (EVT)
- CUSUM Change Point Detection

Machine Learning Models:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoder Neural Network
- DBSCAN Clustering
- Gaussian Mixture Models
- Matrix Profile (STUMPY)

Cyclical/Mean Reversion Models:
- Fourier Transform Analysis
- Hurst Exponent
- Ornstein-Uhlenbeck Process
- Hidden Markov Models
- Wavelet Analysis
- Kalman Filter

Sentiment Analysis:
- Social Media Sentiment
- News Sentiment
- Search Interest Trends
- Put/Call Ratio Analysis
- VIX Analysis

Technical Indicators:
- RSI, MACD, Stochastic
- Bollinger Bands, Keltner Channels
- ADX, Ichimoku Cloud
- OBV, MFI, VWAP
- And 30+ additional indicators

OUTPUTS GENERATED
-----------------
Data Files:
- {output_dir}/data/{ticker}_raw_data.csv
- {output_dir}/data/{ticker}_all_features.csv
- {output_dir}/data/{ticker}_signals.csv
- {output_dir}/data/rally_predictions.csv
- {output_dir}/data/equity_curve.csv
- {output_dir}/data/event_study.csv

Charts:
- {output_dir}/charts/price_signals.png
- {output_dir}/charts/anomaly_heatmap.png
- {output_dir}/charts/model_comparison.png
- {output_dir}/charts/backtest_results.png
- {output_dir}/charts/signal_analysis.png

================================================================================
                              END OF REPORT
================================================================================
"""
    
    # Save summary report
    with open(f"{output_dir}/ANALYSIS_REPORT.txt", 'w') as f:
        f.write(summary_report)
    
    logger.info(summary_report)
    logger.info(f"\nAll outputs saved to: {output_dir}/")
    logger.info("Analysis complete!")
    
    return {
        'data': data,
        'features': all_features,
        'signals': signals,
        'backtest_results': results,
        'predictions': predictions,
        'metrics': metrics
    }


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run analysis
    results = run_full_analysis(
        ticker="RKLB",
        start_date="2021-08-25",
        output_dir="reports"
    )
    
    print("\n" + "="*60)
    print("Analysis complete! Check the 'reports' directory for outputs.")
    print("="*60)
