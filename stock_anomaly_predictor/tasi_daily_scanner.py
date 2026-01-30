#!/usr/bin/env python3
"""
TASI Daily Market Scanner
========================
Production-grade daily scanner for Saudi Stock Exchange (Tadawul)

Features:
- Scans ALL TASI listed stocks
- Applies screening rules
- Generates signals for qualified stocks
- Sends email report with actionable signals

Run daily at 7:00 PM KSA (after market close)
Market hours: Sunday-Thursday, 10:00 AM - 3:00 PM KSA

Author: Anomaly Prediction System
"""

import os
import sys
import warnings
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import pytz
import json
import traceback

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf

# Import our modules
from models.stock_screener import StockSuitabilityScreener
from models.statistical_anomaly import StatisticalAnomalyDetector
from models.ml_anomaly import MLAnomalyDetector
from models.cyclical_models import CyclicalAnalyzer
from models.signal_aggregator import SignalAggregator
from analysis.sentiment import SentimentAnalyzer

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'email': {
        'recipient': 'n.aljudayi@gmail.com',
        'sender': 'tasi.scanner@anomaly-predictor.com',
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        # Note: For production, use environment variables or secure vault
        'username': os.environ.get('SMTP_USERNAME', ''),
        'password': os.environ.get('SMTP_PASSWORD', ''),
    },
    'analysis': {
        'lookback_days': 365 * 3,  # 3 years of history
        'min_data_days': 200,
        'screening_threshold': 50,  # Minimum score for analysis
        'signal_lookback_days': 5,  # Days to look back for recent signals
    },
    'timezone': 'Asia/Riyadh',
}

# ============================================================================
# COMPLETE TASI STOCK UNIVERSE
# ============================================================================
# All major TASI stocks organized by sector

TASI_STOCKS = {
    # Energy Sector
    'energy': [
        '2222.SR',  # Saudi Aramco
        '2030.SR',  # Saudi Kayan Petrochemical
        '2310.SR',  # Sipchem
        '2330.SR',  # Advanced Petrochemical
        '2380.SR',  # Petro Rabigh
        '2060.SR',  # Tasnee
        '4030.SR',  # Bahri (National Shipping)
        '2210.SR',  # Nama Chemicals
        '2170.SR',  # Alujain
    ],
    
    # Banking & Financial Services
    'banking': [
        '1180.SR',  # Al Rajhi Bank
        '1010.SR',  # Riyad Bank
        '1050.SR',  # Banque Saudi Fransi
        '1060.SR',  # Saudi Awwal Bank (SAB)
        '1080.SR',  # Arab National Bank
        '1120.SR',  # Al Jazira Bank
        '1140.SR',  # Bank Albilad
        '1150.SR',  # Alinma Bank
        '1020.SR',  # Bank AlJazira
        '1030.SR',  # Saudi Investment Bank
    ],
    
    # Insurance
    'insurance': [
        '8010.SR',  # Tawuniya
        '8020.SR',  # Malath Insurance
        '8030.SR',  # MEDGULF
        '8040.SR',  # SABB Takaful
        '8050.SR',  # Salama
        '8060.SR',  # Walaa Insurance
        '8070.SR',  # Arabia Insurance
        '8100.SR',  # SAICO
        '8120.SR',  # Gulf Union
        '8150.SR',  # ACIG
        '8160.SR',  # Alahli Takaful
        '8170.SR',  # Alinma Tokio
        '8180.SR',  # Al Sagr Insurance
        '8190.SR',  # United Cooperative
        '8200.SR',  # Saudi Re
        '8210.SR',  # Bupa Arabia
        '8230.SR',  # Al Rajhi Takaful
        '8240.SR',  # CHUBB Arabia
        '8250.SR',  # AXA Cooperative
        '8260.SR',  # Gulf General
        '8270.SR',  # Buruj Insurance
        '8280.SR',  # Al Alamiya
        '8300.SR',  # Wataniya Insurance
        '8310.SR',  # Amana Insurance
        '8311.SR',  # Solidarity Saudi Takaful
    ],
    
    # Materials & Chemicals
    'materials': [
        '2010.SR',  # SABIC
        '2020.SR',  # SAFCO
        '2250.SR',  # Saudi Industrial Investment
        '2290.SR',  # Yanbu National Petrochemical
        '2300.SR',  # Saudi Pharmaceutical
        '3010.SR',  # Arabian Cement
        '3020.SR',  # Yamama Cement
        '3030.SR',  # Saudi Cement
        '3040.SR',  # Saudi Vitrified Clay Pipes
        '3050.SR',  # Southern Cement
        '3060.SR',  # Yanbu Cement
        '3080.SR',  # Eastern Cement
        '3090.SR',  # Tabuk Cement
        '3091.SR',  # Al Jouf Cement
        '3001.SR',  # Hail Cement
        '3002.SR',  # Najran Cement
        '3003.SR',  # City Cement
        '3004.SR',  # Northern Cement
        '3005.SR',  # Umm Al-Qura Cement
        '2040.SR',  # Saudi Ceramic
        '2150.SR',  # Saudi Transport
        '2180.SR',  # Filing & Packing
        '2200.SR',  # Arabian Pipes
        '1211.SR',  # Ma'aden (Saudi Arabian Mining)
        '1212.SR',  # Astra Industrial
        '1301.SR',  # United Wire Factories
        '1304.SR',  # Al-Babtain Power
        '1320.SR',  # Saudi Steel Pipe
        '1321.SR',  # APC
        '1322.SR',  # Amak
    ],
    
    # Real Estate
    'real_estate': [
        '4300.SR',  # Dar Al Arkan
        '4310.SR',  # Emaar Economic City
        '4320.SR',  # Al Andalus Property
        '4250.SR',  # Jabal Omar
        '4220.SR',  # Emaar Properties
        '4230.SR',  # Red Sea International
        '4240.SR',  # Fawaz Alhokair
        '4150.SR',  # Taiba Holding
        '4100.SR',  # Makkah Construction
        '4090.SR',  # Taiba Investment
        '4080.SR',  # Saudi Home Loans
    ],
    
    # Retail & Consumer
    'retail': [
        '4190.SR',  # Jarir Marketing
        '4001.SR',  # Abdullah Al Othaim Markets
        '4003.SR',  # Extra (United Electronics)
        '4006.SR',  # Fawaz Al Hokair
        '4050.SR',  # Saudi Automotive Services
        '4051.SR',  # Bawan Company
        '4061.SR',  # Anaam International
        '4070.SR',  # Tihama Advertising
        '4071.SR',  # Raydan Food
        '4180.SR',  # Aldrees Petroleum
        '4200.SR',  # Saudi Real Estate
        '4260.SR',  # Budget Saudi
        '4261.SR',  # Mohammed Al-Mojil Group (MMG)
        '4262.SR',  # Lumi Rental
        '4270.SR',  # Saudi Marketing
        '4291.SR',  # Remal
        '4292.SR',  # SACO
    ],
    
    # Telecom & IT
    'telecom': [
        '7010.SR',  # STC (Saudi Telecom)
        '7020.SR',  # Etihad Etisalat (Mobily)
        '7030.SR',  # Zain KSA
        '7040.SR',  # Saudi Net
        '7200.SR',  # Solutions by STC
        '7201.SR',  # Elm Company
        '7202.SR',  # Geidea
        '7203.SR',  # Perfect Presentation
        '7204.SR',  # Tawasul
    ],
    
    # Healthcare
    'healthcare': [
        '4002.SR',  # Mouwasat Medical Services
        '4004.SR',  # Dallah Healthcare
        '4005.SR',  # Care (Medical Care Group)
        '4007.SR',  # Al Hammadi
        '4009.SR',  # Middle East Healthcare
        '4013.SR',  # Nahdi Medical
        '4014.SR',  # Al Dawaa Medical
    ],
    
    # Food & Beverages
    'food': [
        '2050.SR',  # Savola Group
        '2100.SR',  # Wafrah for Industry
        '2270.SR',  # Saudia Dairy & Foodstuff
        '2280.SR',  # Almarai
        '6001.SR',  # Halwani Bros
        '6002.SR',  # Herfy Food Services
        '6004.SR',  # Catering (Saudi Airlines)
        '6010.SR',  # NADEC
        '6012.SR',  # Americana Restaurants
        '6013.SR',  # Theeb Rent
        '6014.SR',  # Al Taiseer
        '6015.SR',  # Tamimi Markets
        '6020.SR',  # Aljouf Agriculture
        '6040.SR',  # Saudi Fisheries
        '6050.SR',  # Saudi Grain
        '6060.SR',  # Sharqiya Development
        '6070.SR',  # Alessa Industries
        '6090.SR',  # Jazadco
        '4031.SR',  # Saudi Airlines Catering
    ],
    
    # Industrial & Manufacturing
    'industrial': [
        '2070.SR',  # Saudi Chemical
        '2080.SR',  # GACO
        '2090.SR',  # National Gypsum
        '2110.SR',  # Saudi Cable
        '2120.SR',  # SATORP
        '2130.SR',  # SIIG
        '2160.SR',  # Saudi Vitrified
        '2190.SR',  # Saudi Industrial Services
        '2220.SR',  # NASCI
        '2230.SR',  # Chemical Holding
        '2240.SR',  # ZOUJAJ
        '2320.SR',  # Al Babtain
        '2340.SR',  # Yansab
        '2350.SR',  # NAMA
        '2360.SR',  # SVCP
        '2370.SR',  # Middle East Specialized
        '1210.SR',  # BCI
        '1213.SR',  # NSCSA
        '1214.SR',  # Shaker
        '1302.SR',  # Bawan
        '1303.SR',  # Electric
        '2001.SR',  # Chemanol
        '2002.SR',  # Petrochem
        '2003.SR',  # SPM
    ],
    
    # Utilities
    'utilities': [
        '5110.SR',  # Saudi Electricity
        '2082.SR',  # ACWA Power
        '2083.SR',  # Marafiq
    ],
    
    # Diversified / Holding
    'holding': [
        '4280.SR',  # Kingdom Holding
        '4020.SR',  # Saudi Real Estate
        '4010.SR',  # Saudi Investment
        '4210.SR',  # Al Khaleej Training
        '1201.SR',  # Takween
        '1202.SR',  # MEPCO
        '4040.SR',  # Saudi Ground Services
        '4160.SR',  # Thimar
        '4170.SR',  # Dallah
    ],
}

# Flatten to single list
ALL_TASI_STOCKS = []
for sector, stocks in TASI_STOCKS.items():
    ALL_TASI_STOCKS.extend(stocks)

# Remove duplicates while preserving order
ALL_TASI_STOCKS = list(dict.fromkeys(ALL_TASI_STOCKS))


class TASIDailyScanner:
    """Daily scanner for TASI market"""
    
    def __init__(self):
        self.ksa_tz = pytz.timezone(CONFIG['timezone'])
        self.scan_date = datetime.now(self.ksa_tz)
        self.results = {
            'scan_time': self.scan_date.isoformat(),
            'stocks_scanned': 0,
            'stocks_qualified': 0,
            'signals_generated': 0,
            'screening_results': [],
            'active_signals': [],
            'errors': []
        }
        
        # Ensure output directory exists
        os.makedirs('output/daily_scans', exist_ok=True)
    
    def log(self, message):
        """Print with timestamp"""
        timestamp = datetime.now(self.ksa_tz).strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def get_market_benchmark(self):
        """Get market benchmark data"""
        self.log("Fetching market benchmark (S&P 500)...")
        try:
            market = yf.Ticker("^GSPC")
            start_date = (datetime.now() - timedelta(days=CONFIG['analysis']['lookback_days'])).strftime('%Y-%m-%d')
            self.market_df = market.history(start=start_date)
            self.market_returns = self.market_df['Close'].pct_change().dropna()
            return True
        except Exception as e:
            self.log(f"Error fetching benchmark: {e}")
            self.results['errors'].append(f"Benchmark error: {str(e)}")
            return False
    
    def screen_stock(self, ticker):
        """Screen a single stock"""
        try:
            stock = yf.Ticker(ticker)
            start_date = (datetime.now() - timedelta(days=CONFIG['analysis']['lookback_days'])).strftime('%Y-%m-%d')
            df = stock.history(start=start_date)
            
            if len(df) < CONFIG['analysis']['min_data_days']:
                return None, f"Insufficient data ({len(df)} days)"
            
            df.columns = [c.lower() for c in df.columns]
            df.name = ticker
            
            # Get company info
            try:
                info = stock.info
                company_name = info.get('shortName', ticker)[:40]
            except:
                company_name = ticker
            
            # Run screening
            screener = StockSuitabilityScreener(df, self.market_df)
            score = screener.analyze(self.market_returns)
            
            return {
                'ticker': ticker,
                'name': company_name,
                'score': score.overall_score,
                'recommendation': score.recommendation,
                'momentum': score.momentum_score,
                'trend': score.trend_strength_score,
                'beta': score.beta,
                'hurst': score.hurst_exponent,
                'volatility': score.volatility,
                'df': df
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def analyze_stock(self, ticker, df, screen_result):
        """Run full analysis on a qualified stock"""
        try:
            df['returns'] = df['close'].pct_change()
            
            # Calculate forward returns for validation
            for days in [1, 3, 5, 10, 20]:
                df[f'fwd_ret_{days}d'] = df['close'].shift(-days) / df['close'] - 1
            
            # Statistical Anomaly Detection
            stat_detector = StatisticalAnomalyDetector(df)
            stat_anomalies = stat_detector.run_all_detectors()
            
            # ML Anomaly Detection
            ml_detector = MLAnomalyDetector(df)
            ml_anomalies = ml_detector.run_all_detectors(include_deep_learning=False)
            
            # Cyclical Analysis
            cyclical_analyzer = CyclicalAnalyzer(df)
            cyclical_signals = cyclical_analyzer.run_all_analysis()
            
            # Sentiment (simulated for now)
            sentiment_analyzer = SentimentAnalyzer(ticker)
            sentiment_signals = sentiment_analyzer.run_full_analysis(df)
            
            # Signal Aggregation
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
            
            # Get recent signals (last N days)
            lookback = CONFIG['analysis']['signal_lookback_days']
            recent_date = df.index[-1] - timedelta(days=lookback)
            
            recent_signals = trade_signals[
                (trade_signals.index >= recent_date) & 
                (trade_signals['actionable_signal'] == 1)
            ]
            
            signals = []
            for date, row in recent_signals.iterrows():
                price_idx = df.index.get_loc(date)
                current_price = df['close'].iloc[-1]
                signal_price = df['close'].iloc[price_idx]
                change_since_signal = (current_price / signal_price - 1) * 100
                
                signals.append({
                    'ticker': ticker,
                    'company': screen_result['name'],
                    'date': date.strftime('%Y-%m-%d'),
                    'signal_price': round(signal_price, 2),
                    'current_price': round(current_price, 2),
                    'change_pct': round(change_since_signal, 2),
                    'rally_probability': round(row.get('rally_probability', 0) * 100, 1),
                    'signal_confidence': round(row.get('signal_confidence', 0) * 100, 1),
                    'anomaly_intensity': round(row.get('anomaly_intensity', 0), 2),
                    'screening_score': screen_result['score'],
                    'recommendation': screen_result['recommendation']
                })
            
            # Check if there's a signal TODAY
            today = df.index[-1]
            has_today_signal = len(trade_signals[
                (trade_signals.index == today) & 
                (trade_signals['actionable_signal'] == 1)
            ]) > 0
            
            return signals, has_today_signal
            
        except Exception as e:
            self.log(f"  Analysis error: {str(e)[:50]}")
            return [], False
    
    def run_scan(self):
        """Run the complete daily scan"""
        self.log("=" * 70)
        self.log("TASI DAILY MARKET SCAN")
        self.log(f"Date: {self.scan_date.strftime('%Y-%m-%d %H:%M:%S KSA')}")
        self.log("=" * 70)
        
        # Get benchmark
        if not self.get_market_benchmark():
            return False
        
        # Phase 1: Screen all stocks
        self.log(f"\nPHASE 1: Screening {len(ALL_TASI_STOCKS)} TASI stocks...")
        self.log("-" * 70)
        
        screened = []
        failed = []
        
        for i, ticker in enumerate(ALL_TASI_STOCKS):
            if (i + 1) % 20 == 0:
                self.log(f"  Progress: {i+1}/{len(ALL_TASI_STOCKS)} stocks screened...")
            
            result, error = self.screen_stock(ticker)
            
            if result:
                screened.append(result)
                self.results['screening_results'].append({
                    'ticker': result['ticker'],
                    'name': result['name'],
                    'score': result['score'],
                    'recommendation': result['recommendation']
                })
            else:
                failed.append((ticker, error))
        
        self.results['stocks_scanned'] = len(screened)
        self.log(f"\nScreened: {len(screened)} stocks | Failed: {len(failed)} stocks")
        
        # Filter qualified stocks
        qualified = [s for s in screened if s['score'] >= CONFIG['analysis']['screening_threshold']]
        self.results['stocks_qualified'] = len(qualified)
        
        self.log(f"Qualified for analysis (score >= {CONFIG['analysis']['screening_threshold']}): {len(qualified)} stocks")
        
        # Phase 2: Analyze qualified stocks
        self.log(f"\nPHASE 2: Analyzing {len(qualified)} qualified stocks...")
        self.log("-" * 70)
        
        all_signals = []
        today_signals = []
        
        for i, stock in enumerate(sorted(qualified, key=lambda x: x['score'], reverse=True)):
            ticker = stock['ticker']
            self.log(f"  [{i+1}/{len(qualified)}] {ticker} (Score: {stock['score']:.1f})...")
            
            signals, has_today = self.analyze_stock(ticker, stock['df'], stock)
            
            if signals:
                all_signals.extend(signals)
                if has_today:
                    today_signals.extend([s for s in signals if s['date'] == stock['df'].index[-1].strftime('%Y-%m-%d')])
        
        self.results['active_signals'] = all_signals
        self.results['signals_generated'] = len(all_signals)
        
        self.log(f"\nTotal signals in last {CONFIG['analysis']['signal_lookback_days']} days: {len(all_signals)}")
        self.log(f"Signals TODAY: {len(today_signals)}")
        
        # Save results
        self.save_results()
        
        return True
    
    def save_results(self):
        """Save scan results to files"""
        date_str = self.scan_date.strftime('%Y%m%d')
        
        # Save JSON
        json_file = f"output/daily_scans/tasi_scan_{date_str}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"Saved: {json_file}")
        
        # Save signals CSV
        if self.results['active_signals']:
            signals_df = pd.DataFrame(self.results['active_signals'])
            csv_file = f"output/daily_scans/tasi_signals_{date_str}.csv"
            signals_df.to_csv(csv_file, index=False)
            self.log(f"Saved: {csv_file}")
        
        # Save screening CSV
        if self.results['screening_results']:
            screen_df = pd.DataFrame(self.results['screening_results'])
            screen_df = screen_df.sort_values('score', ascending=False)
            screen_csv = f"output/daily_scans/tasi_screening_{date_str}.csv"
            screen_df.to_csv(screen_csv, index=False)
            self.log(f"Saved: {screen_csv}")
    
    def generate_email_report(self):
        """Generate HTML email report"""
        
        # Get today's date in KSA
        today_str = self.scan_date.strftime('%A, %B %d, %Y')
        
        # Sort signals
        signals = sorted(
            self.results['active_signals'],
            key=lambda x: (x['date'], -x['rally_probability']),
            reverse=True
        )
        
        # Separate today's signals
        today_date = self.scan_date.strftime('%Y-%m-%d')
        today_signals = [s for s in signals if s['date'] == today_date]
        recent_signals = [s for s in signals if s['date'] != today_date][:20]
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1a5f2a 0%, #2d8a3e 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .summary {{ display: flex; justify-content: space-around; padding: 20px; background: #f8f9fa; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #1a5f2a; }}
        .stat-label {{ color: #666; font-size: 12px; text-transform: uppercase; }}
        .section {{ padding: 20px; }}
        .section-title {{ color: #1a5f2a; border-bottom: 2px solid #1a5f2a; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #1a5f2a; color: white; padding: 12px 8px; text-align: left; font-size: 12px; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
        tr:hover {{ background: #f5f5f5; }}
        .signal-new {{ background: #e8f5e9 !important; }}
        .ticker {{ font-weight: bold; color: #1a5f2a; }}
        .prob-high {{ color: #2e7d32; font-weight: bold; }}
        .prob-medium {{ color: #f57c00; }}
        .change-positive {{ color: #2e7d32; }}
        .change-negative {{ color: #c62828; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
        .badge-excellent {{ background: #2e7d32; color: white; }}
        .badge-good {{ background: #558b2f; color: white; }}
        .badge-moderate {{ background: #f57c00; color: white; }}
        .alert {{ background: #fff3e0; border-left: 4px solid #f57c00; padding: 15px; margin: 15px 0; }}
        .alert-success {{ background: #e8f5e9; border-left-color: #2e7d32; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; border-radius: 0 0 10px 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇸🇦 TASI Daily Signal Report</h1>
            <p>{today_str} | Saudi Stock Exchange (Tadawul)</p>
        </div>
        
        <div class="summary">
            <div class="stat">
                <div class="stat-value">{self.results['stocks_scanned']}</div>
                <div class="stat-label">Stocks Scanned</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.results['stocks_qualified']}</div>
                <div class="stat-label">Qualified</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(today_signals)}</div>
                <div class="stat-label">Today's Signals</div>
            </div>
            <div class="stat">
                <div class="stat-value">{self.results['signals_generated']}</div>
                <div class="stat-label">Total Active</div>
            </div>
        </div>
"""
        
        # Today's signals section
        if today_signals:
            html += """
        <div class="section">
            <h2 class="section-title">🔔 NEW SIGNALS TODAY</h2>
            <div class="alert alert-success">
                <strong>Action Required:</strong> The following stocks triggered buy signals today.
            </div>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Price (SAR)</th>
                    <th>Probability</th>
                    <th>Confidence</th>
                    <th>Score</th>
                </tr>
"""
            for s in today_signals:
                prob_class = 'prob-high' if s['rally_probability'] >= 60 else 'prob-medium'
                badge_class = 'badge-excellent' if s['recommendation'] == 'EXCELLENT' else ('badge-good' if s['recommendation'] == 'GOOD' else 'badge-moderate')
                
                html += f"""
                <tr class="signal-new">
                    <td class="ticker">{s['ticker']}</td>
                    <td>{s['company'][:25]}</td>
                    <td>{s['current_price']:.2f}</td>
                    <td class="{prob_class}">{s['rally_probability']:.0f}%</td>
                    <td>{s['signal_confidence']:.0f}%</td>
                    <td><span class="badge {badge_class}">{s['screening_score']:.0f}</span></td>
                </tr>
"""
            html += """
            </table>
        </div>
"""
        else:
            html += """
        <div class="section">
            <h2 class="section-title">🔔 TODAY'S SIGNALS</h2>
            <div class="alert">
                No new signals triggered today. Continue monitoring recent active signals below.
            </div>
        </div>
"""
        
        # Recent active signals
        if recent_signals:
            html += """
        <div class="section">
            <h2 class="section-title">📊 Recent Active Signals (Last 5 Days)</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Signal Price</th>
                    <th>Current</th>
                    <th>Change</th>
                    <th>Prob</th>
                </tr>
"""
            for s in recent_signals:
                change_class = 'change-positive' if s['change_pct'] >= 0 else 'change-negative'
                change_sign = '+' if s['change_pct'] >= 0 else ''
                
                html += f"""
                <tr>
                    <td>{s['date']}</td>
                    <td class="ticker">{s['ticker']}</td>
                    <td>{s['company'][:20]}</td>
                    <td>{s['signal_price']:.2f}</td>
                    <td>{s['current_price']:.2f}</td>
                    <td class="{change_class}">{change_sign}{s['change_pct']:.1f}%</td>
                    <td>{s['rally_probability']:.0f}%</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""
        
        # Top screened stocks
        top_screened = sorted(self.results['screening_results'], key=lambda x: x['score'], reverse=True)[:15]
        html += """
        <div class="section">
            <h2 class="section-title">📈 Top Screened Stocks</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Score</th>
                    <th>Rating</th>
                </tr>
"""
        for i, s in enumerate(top_screened, 1):
            badge_class = 'badge-excellent' if s['recommendation'] == 'EXCELLENT' else ('badge-good' if s['recommendation'] == 'GOOD' else 'badge-moderate')
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td class="ticker">{s['ticker']}</td>
                    <td>{s['name'][:25]}</td>
                    <td><strong>{s['score']:.1f}</strong></td>
                    <td><span class="badge {badge_class}">{s['recommendation']}</span></td>
                </tr>
"""
        html += """
            </table>
        </div>
"""
        
        # Footer
        html += f"""
        <div class="footer">
            <p>Generated by TASI Anomaly Prediction System</p>
            <p>Scan completed at {self.scan_date.strftime('%H:%M:%S KSA')} | Next scan: Tomorrow at 19:00 KSA</p>
            <p style="color: #999; font-size: 10px;">
                Disclaimer: This is not financial advice. Always conduct your own research before making investment decisions.
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def send_email(self):
        """Send email report"""
        self.log("\nSending email report...")
        
        # Generate HTML report
        html_content = self.generate_email_report()
        
        # Save HTML locally
        date_str = self.scan_date.strftime('%Y%m%d')
        html_file = f"output/daily_scans/tasi_report_{date_str}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        self.log(f"Saved HTML report: {html_file}")
        
        # Try to send email
        email_config = CONFIG['email']
        
        if not email_config['username'] or not email_config['password']:
            self.log("⚠️  Email credentials not configured. Report saved locally.")
            self.log(f"   To enable email, set SMTP_USERNAME and SMTP_PASSWORD environment variables")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🇸🇦 TASI Daily Signals - {self.scan_date.strftime('%Y-%m-%d')}"
            msg['From'] = email_config['sender']
            msg['To'] = email_config['recipient']
            
            # Plain text version
            text_content = f"""
TASI Daily Signal Report - {self.scan_date.strftime('%Y-%m-%d')}

Stocks Scanned: {self.results['stocks_scanned']}
Qualified: {self.results['stocks_qualified']}
Active Signals: {self.results['signals_generated']}

See attached HTML report for details.
"""
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send
            context = ssl.create_default_context()
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(email_config['username'], email_config['password'])
                server.sendmail(email_config['sender'], email_config['recipient'], msg.as_string())
            
            self.log(f"✓ Email sent to {email_config['recipient']}")
            return True
            
        except Exception as e:
            self.log(f"✗ Email failed: {str(e)}")
            self.results['errors'].append(f"Email error: {str(e)}")
            return False


def main():
    """Main entry point"""
    scanner = TASIDailyScanner()
    
    try:
        # Run scan
        success = scanner.run_scan()
        
        if success:
            # Send email report
            scanner.send_email()
            
            # Print summary
            print("\n" + "=" * 70)
            print("SCAN COMPLETE")
            print("=" * 70)
            print(f"  Stocks scanned:    {scanner.results['stocks_scanned']}")
            print(f"  Stocks qualified:  {scanner.results['stocks_qualified']}")
            print(f"  Signals generated: {scanner.results['signals_generated']}")
            
            if scanner.results['active_signals']:
                print(f"\n  Active Signals:")
                for s in sorted(scanner.results['active_signals'], key=lambda x: x['date'], reverse=True)[:10]:
                    print(f"    {s['date']} | {s['ticker']:<10} | {s['company'][:20]:<20} | "
                          f"Prob: {s['rally_probability']:.0f}% | Price: {s['current_price']:.2f} SAR")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
