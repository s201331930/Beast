#!/usr/bin/env python3
"""
TASI Detailed Signal Analysis

Generate detailed signal-by-signal breakdown for Saudi stocks:
- Signal date
- Price at signal
- Signal strength
- Forward returns
- Hit/miss classification
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf

# Import our modules
from models.statistical_anomaly import StatisticalAnomalyDetector
from models.ml_anomaly import MLAnomalyDetector
from models.cyclical_models import CyclicalAnalyzer
from models.signal_aggregator import SignalAggregator
from analysis.sentiment import SentimentAnalyzer

print("=" * 100)
print("TASI DETAILED SIGNAL ANALYSIS")
print("Signal-by-Signal Breakdown for Saudi Stocks")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Load the analysis results to get the top stocks
analysis_df = pd.read_csv('output/tasi_analysis_results.csv')

# Focus on the recommended stocks (those with good performance)
recommended_tickers = [
    '3020.SR',   # Yamama Cement - Best 10D hit
    '1150.SR',   # Alinma Bank
    '7020.SR',   # Etihad Etisalat (Mobily)
    '1180.SR',   # Saudi National Bank
    '4300.SR',   # Dar Alarkan Real Estate
    '2222.SR',   # Saudi Aramco
    '1140.SR',   # Bank Albilad
    '4009.SR',   # Middle East Healthcare
    '1320.SR',   # Saudi Steel Pipe
    '1080.SR',   # Arab National Bank
    '1050.SR',   # Banque Saudi Fransi
]

START_DATE = "2021-08-25"

all_signals = []

for ticker in recommended_tickers:
    print(f"\n{'='*100}")
    print(f"ANALYZING SIGNALS: {ticker}")
    print(f"{'='*100}")
    
    try:
        # Get company info
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            company_name = info.get('shortName', ticker)[:40]
        except:
            company_name = ticker
        
        print(f"Company: {company_name}")
        
        # Fetch data
        df = stock.history(start=START_DATE)
        if len(df) < 200:
            print(f"Insufficient data: {len(df)} days")
            continue
        
        df.columns = [c.lower() for c in df.columns]
        df['returns'] = df['close'].pct_change()
        
        # Calculate forward returns
        for days in [1, 3, 5, 10, 20]:
            df[f'fwd_ret_{days}d'] = df['close'].shift(-days) / df['close'] - 1
        
        print(f"Data: {len(df)} days, from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current Price: {df['close'].iloc[-1]:.2f} SAR")
        
        # Run analysis pipeline
        print("Running analysis pipeline...", end=" ")
        
        # Statistical
        stat_detector = StatisticalAnomalyDetector(df)
        stat_anomalies = stat_detector.run_all_detectors()
        
        # ML
        ml_detector = MLAnomalyDetector(df)
        ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)
        
        # Cyclical
        cyclical_analyzer = CyclicalAnalyzer(df)
        cyclical_signals = cyclical_analyzer.run_all_analysis()
        
        # Sentiment
        sentiment_analyzer = SentimentAnalyzer(ticker)
        sentiment_signals = sentiment_analyzer.run_full_analysis(df)
        
        # Aggregate
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
        
        print("Done")
        
        # Extract actionable signals
        signals = trade_signals[trade_signals['actionable_signal'] == 1].copy()
        
        if len(signals) == 0:
            print("No actionable signals generated")
            continue
        
        print(f"\nTotal Actionable Signals: {len(signals)}")
        
        # Merge with price data
        signals = signals.join(df[['close', 'volume', 'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d']])
        
        # Print signal details
        print(f"\n{'-'*100}")
        print(f"SIGNAL DETAILS FOR {ticker}")
        print(f"{'-'*100}")
        
        print(f"\n{'Date':<12} {'Price':>8} {'Prob':>6} {'Conf':>6} {'Anom':>6} | "
              f"{'1D':>7} {'3D':>7} {'5D':>7} {'10D':>7} {'20D':>7} | {'Result':<8}")
        print("-" * 100)
        
        signal_count = 0
        hits_5d = 0
        hits_10d = 0
        
        for date, row in signals.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            price = row.get('close', 0)
            prob = row.get('rally_probability', 0)
            conf = row.get('signal_confidence', 0)
            anom = row.get('anomaly_intensity', 0)
            
            ret_1d = row.get('fwd_ret_1d', np.nan)
            ret_3d = row.get('fwd_ret_3d', np.nan)
            ret_5d = row.get('fwd_ret_5d', np.nan)
            ret_10d = row.get('fwd_ret_10d', np.nan)
            ret_20d = row.get('fwd_ret_20d', np.nan)
            
            # Determine hit/miss (5% gain within 10 days)
            is_hit_5d = ret_5d >= 0.02 if pd.notna(ret_5d) else None
            is_hit_10d = ret_10d >= 0.03 if pd.notna(ret_10d) else None
            
            if is_hit_5d:
                hits_5d += 1
            if is_hit_10d:
                hits_10d += 1
            
            if pd.isna(ret_10d):
                result = "PENDING"
            elif ret_10d >= 0.05:
                result = "BIG WIN"
            elif ret_10d >= 0.02:
                result = "WIN"
            elif ret_10d >= -0.02:
                result = "FLAT"
            else:
                result = "LOSS"
            
            signal_count += 1
            
            # Format returns
            def fmt_ret(r):
                if pd.isna(r):
                    return "   -  "
                color = "" 
                return f"{r:+6.1%}"
            
            print(f"{date_str:<12} {price:>8.2f} {prob:>5.1%} {conf:>5.1%} {anom:>5.2f} | "
                  f"{fmt_ret(ret_1d)} {fmt_ret(ret_3d)} {fmt_ret(ret_5d)} {fmt_ret(ret_10d)} {fmt_ret(ret_20d)} | {result:<8}")
            
            # Store for summary
            all_signals.append({
                'ticker': ticker,
                'company': company_name,
                'date': date_str,
                'price': price,
                'rally_probability': prob,
                'signal_confidence': conf,
                'anomaly_intensity': anom,
                'return_1d': ret_1d,
                'return_3d': ret_3d,
                'return_5d': ret_5d,
                'return_10d': ret_10d,
                'return_20d': ret_20d,
                'result': result
            })
        
        # Summary for this stock
        valid_10d = signals['fwd_ret_10d'].dropna()
        print(f"\n{'─'*100}")
        print(f"SUMMARY FOR {ticker}:")
        print(f"  Total Signals: {signal_count}")
        print(f"  5D Hit Rate (≥2%):  {hits_5d}/{len(valid_10d)} = {hits_5d/len(valid_10d):.1%}" if len(valid_10d) > 0 else "  N/A")
        print(f"  10D Hit Rate (≥3%): {hits_10d}/{len(valid_10d)} = {hits_10d/len(valid_10d):.1%}" if len(valid_10d) > 0 else "  N/A")
        print(f"  Avg 10D Return: {valid_10d.mean():.2%}" if len(valid_10d) > 0 else "  N/A")
        print(f"  Best 10D Return: {valid_10d.max():.2%}" if len(valid_10d) > 0 else "  N/A")
        print(f"  Worst 10D Return: {valid_10d.min():.2%}" if len(valid_10d) > 0 else "  N/A")
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")

# ============================================================================
# CONSOLIDATED SIGNAL TABLE
# ============================================================================
print("\n" + "=" * 100)
print("CONSOLIDATED SIGNAL TABLE - ALL TASI STOCKS")
print("=" * 100)

signals_df = pd.DataFrame(all_signals)
signals_df = signals_df.sort_values('date')

# Save to CSV
signals_df.to_csv('output/tasi_signal_details.csv', index=False)
print(f"\nDetailed signals saved to: output/tasi_signal_details.csv")

# Summary statistics
print("\n" + "-" * 100)
print("SIGNAL SUMMARY BY STOCK")
print("-" * 100)

print(f"\n{'Ticker':<12} {'Company':<25} {'Signals':>8} {'Avg 10D':>9} {'Hit Rate':>10} {'Best':>9} {'Worst':>9}")
print("-" * 100)

for ticker in signals_df['ticker'].unique():
    stock_signals = signals_df[signals_df['ticker'] == ticker]
    valid = stock_signals['return_10d'].dropna()
    
    company = stock_signals['company'].iloc[0][:22] if len(stock_signals) > 0 else ''
    num_signals = len(stock_signals)
    avg_ret = valid.mean() if len(valid) > 0 else 0
    hit_rate = (valid >= 0.03).sum() / len(valid) if len(valid) > 0 else 0
    best = valid.max() if len(valid) > 0 else 0
    worst = valid.min() if len(valid) > 0 else 0
    
    print(f"{ticker:<12} {company:<25} {num_signals:>8} {avg_ret:>+8.2%} {hit_rate:>9.1%} {best:>+8.2%} {worst:>+8.2%}")

# Most recent signals
print("\n" + "-" * 100)
print("MOST RECENT SIGNALS (Last 30 Days)")
print("-" * 100)

recent = signals_df.tail(30)
print(f"\n{'Date':<12} {'Ticker':<12} {'Company':<25} {'Price':>8} {'Prob':>7} {'10D Ret':>9}")
print("-" * 100)

for _, row in recent.iterrows():
    ret_10d = row['return_10d']
    ret_str = f"{ret_10d:+8.2%}" if pd.notna(ret_10d) else "PENDING"
    company = str(row['company'])[:22] if pd.notna(row['company']) else ''
    print(f"{row['date']:<12} {row['ticker']:<12} {company:<25} {row['price']:>8.2f} {row['rally_probability']:>6.1%} {ret_str:>9}")

# Best and worst signals
print("\n" + "-" * 100)
print("TOP 10 BEST SIGNALS (by 10-Day Return)")
print("-" * 100)

valid_signals = signals_df[signals_df['return_10d'].notna()].copy()
best_signals = valid_signals.nlargest(10, 'return_10d')

print(f"\n{'Date':<12} {'Ticker':<12} {'Company':<22} {'Price':>8} {'Prob':>7} {'10D':>8} {'20D':>8}")
print("-" * 100)
for _, row in best_signals.iterrows():
    company = str(row['company'])[:20] if pd.notna(row['company']) else ''
    ret_20d = f"{row['return_20d']:+7.2%}" if pd.notna(row['return_20d']) else "   -   "
    print(f"{row['date']:<12} {row['ticker']:<12} {company:<22} {row['price']:>8.2f} "
          f"{row['rally_probability']:>6.1%} {row['return_10d']:>+7.2%} {ret_20d}")

# Signal frequency by month
print("\n" + "-" * 100)
print("SIGNAL FREQUENCY BY MONTH")
print("-" * 100)

signals_df['month'] = pd.to_datetime(signals_df['date']).dt.to_period('M')
monthly = signals_df.groupby('month').agg({
    'ticker': 'count',
    'return_10d': ['mean', lambda x: (x >= 0.03).sum() / len(x) if len(x) > 0 else 0]
}).round(3)

monthly.columns = ['Signal Count', 'Avg 10D Return', 'Hit Rate']
print(f"\n{monthly.to_string()}")

# ============================================================================
# SIGNAL CALENDAR VIEW
# ============================================================================
print("\n" + "=" * 100)
print("SIGNAL CALENDAR - 2024 & 2025")
print("=" * 100)

# Filter to recent years
signals_df['year'] = pd.to_datetime(signals_df['date']).dt.year
signals_df['month_num'] = pd.to_datetime(signals_df['date']).dt.month

for year in [2024, 2025]:
    year_signals = signals_df[signals_df['year'] == year]
    if len(year_signals) == 0:
        continue
    
    print(f"\n{year}:")
    print("-" * 80)
    
    for month in range(1, 13):
        month_signals = year_signals[year_signals['month_num'] == month]
        if len(month_signals) == 0:
            continue
        
        month_name = datetime(year, month, 1).strftime('%B')
        print(f"\n  {month_name} {year}:")
        
        for _, row in month_signals.iterrows():
            ret_10d = row['return_10d']
            result = "PENDING" if pd.isna(ret_10d) else (f"{ret_10d:+.1%}" if ret_10d else "0.0%")
            indicator = "✓" if (pd.notna(ret_10d) and ret_10d >= 0.03) else ("○" if pd.isna(ret_10d) else "✗")
            day = pd.to_datetime(row['date']).day
            print(f"    {indicator} {day:2d}: {row['ticker']:<12} @ {row['price']:.2f} SAR → {result}")

print("\n" + "=" * 100)
print("TASI SIGNAL ANALYSIS COMPLETE")
print("=" * 100)
print(f"\nTotal signals analyzed: {len(signals_df)}")
print(f"Stocks covered: {signals_df['ticker'].nunique()}")
print(f"Date range: {signals_df['date'].min()} to {signals_df['date'].max()}")
