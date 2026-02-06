#!/usr/bin/env python3
"""
Email Module for Production Scanner
===================================
Sends daily reports via email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List

from production.config import EMAIL, STRATEGY


def send_report(report_text: str, buy_signals: List = None) -> bool:
    """
    Send scanner report via email.
    
    Args:
        report_text: Full text report
        buy_signals: List of StockAnalysis objects with BUY recommendation
    
    Returns:
        True if sent successfully, False otherwise
    """
    if not EMAIL['enabled']:
        print("Email disabled in config")
        return False
    
    # Get credentials from environment
    username = os.environ.get('SMTP_USERNAME', '')
    password = os.environ.get('SMTP_PASSWORD', '')
    
    if not username or not password:
        print("SMTP credentials not found in environment")
        return False
    
    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"TASI Scanner Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = username
    msg['To'] = EMAIL['recipient']
    
    # Plain text version
    msg.attach(MIMEText(report_text, 'plain'))
    
    # HTML version
    html = generate_html_report(report_text, buy_signals)
    msg.attach(MIMEText(html, 'html'))
    
    # Send
    try:
        with smtplib.SMTP(EMAIL['smtp_server'], EMAIL['smtp_port']) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, EMAIL['recipient'], msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


def generate_html_report(report_text: str, buy_signals: List = None) -> str:
    """Generate HTML version of report."""
    
    # Count signals
    n_signals = len(buy_signals) if buy_signals else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            h1 {{ color: #1a5f2a; border-bottom: 2px solid #1a5f2a; padding-bottom: 10px; }}
            h2 {{ color: #2c7a3f; margin-top: 30px; }}
            .regime {{ padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .regime.bull {{ background: #d4edda; border: 1px solid #28a745; }}
            .regime.bear {{ background: #f8d7da; border: 1px solid #dc3545; }}
            .signal-card {{ background: #e8f5e9; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin: 10px 0; }}
            .signal-header {{ font-size: 18px; font-weight: bold; color: #2e7d32; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f0f0f0; }}
            .buy {{ color: #28a745; font-weight: bold; }}
            .watch {{ color: #ffc107; }}
            .avoid {{ color: #dc3545; }}
            .params {{ background: #f8f9fa; padding: 15px; border-radius: 8px; font-family: monospace; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇸🇦 TASI Scanner Report</h1>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="regime {'bull' if n_signals > 0 else 'bear'}">
                <strong>Market Status:</strong> {n_signals} Active Buy Signals
            </div>
    """
    
    if buy_signals:
        html += """
            <h2>📈 Buy Signals</h2>
        """
        for s in buy_signals:
            html += f"""
            <div class="signal-card">
                <div class="signal-header">{s.ticker} - {s.name}</div>
                <table>
                    <tr><td>Current Price</td><td><strong>{s.current_price:.2f} SAR</strong></td></tr>
                    <tr><td>Score</td><td><strong>{s.score:.1f}/100</strong></td></tr>
                    <tr><td>Signal</td><td>{s.signal} (Strength: {s.signal_strength:.0%})</td></tr>
                    <tr><td>Entry Price</td><td>{s.entry_price:.2f} SAR</td></tr>
                    <tr><td>Take Profit</td><td>{s.take_profit:.2f} SAR (+{STRATEGY['take_profit_pct']*100:.0f}%)</td></tr>
                    <tr><td>Trailing Stop</td><td>{s.trailing_stop:.2f} SAR (-{STRATEGY['trailing_stop_pct']*100:.0f}%)</td></tr>
                    <tr><td>Position Size</td><td><strong>{s.position_size_pct:.1f}%</strong> of capital</td></tr>
                </table>
            </div>
            """
    else:
        html += """
            <h2>📊 No Active Buy Signals</h2>
            <p>No stocks currently meet the entry criteria. Continue monitoring the watchlist.</p>
        """
    
    html += f"""
            <h2>⚙️ Strategy Parameters</h2>
            <div class="params">
                <p><strong>Regime Detection:</strong> 50-day Moving Average</p>
                <p><strong>Bull Leverage:</strong> {STRATEGY['leverage_bull']}x | <strong>Bear Leverage:</strong> {STRATEGY['leverage_bear']}x</p>
                <p><strong>Take Profit:</strong> {STRATEGY['take_profit_pct']*100:.0f}% | <strong>Trailing Stop:</strong> {STRATEGY['trailing_stop_pct']*100:.0f}%</p>
                <p><strong>Min Score:</strong> {STRATEGY['min_score']} | <strong>Max Hold:</strong> {STRATEGY['max_holding_days']} days</p>
            </div>
            
            <div class="footer">
                <p>This report was generated automatically by the TASI Scanner system.</p>
                <p>Strategy based on scientific optimization (1,000+ simulations).</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html
