#!/usr/bin/env python3
"""
EMAIL REPORTER FOR COMPLETE SCANNER
===================================
Sends HTML-formatted daily reports with:
1. All trade parameters
2. Performance tracking history
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production.complete_scanner import CompleteScanner, StockPosition


class CompleteEmailer:
    """Sends comprehensive HTML email reports."""
    
    def __init__(self):
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', 587))
        self.email_user = os.environ.get('EMAIL_USER', '')
        self.email_pass = os.environ.get('EMAIL_PASS', '')
        self.recipient = os.environ.get('EMAIL_TO', '')
    
    def generate_html(self, scanner: CompleteScanner, regime: str, leverage: float, 
                      market_price: float, market_ma: float) -> str:
        """Generate comprehensive HTML report."""
        
        regime_color = '#10b981' if regime == 'BULL' else '#ef4444'
        regime_emoji = '🟢' if regime == 'BULL' else '🔴'
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 30px; border-radius: 12px 12px 0 0; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .regime-badge {{ display: inline-block; padding: 8px 20px; border-radius: 20px; font-weight: bold; font-size: 18px; background: {regime_color}; color: white; margin-top: 15px; }}
        .section {{ padding: 25px; border-bottom: 1px solid #e5e7eb; }}
        .section h2 {{ color: #1e40af; margin-top: 0; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px; }}
        .metric {{ background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1e40af; }}
        .metric-label {{ font-size: 12px; color: #64748b; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ background: #1e40af; color: white; padding: 12px; text-align: left; font-size: 13px; }}
        td {{ padding: 12px; border-bottom: 1px solid #e5e7eb; font-size: 13px; }}
        tr:hover {{ background: #f8fafc; }}
        .bull {{ color: #10b981; font-weight: bold; }}
        .bear {{ color: #ef4444; font-weight: bold; }}
        .warning {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .trade-card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .trade-card h3 {{ margin: 0 0 15px 0; color: #1e40af; }}
        .trade-params {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .param {{ background: white; padding: 10px; border-radius: 4px; }}
        .param-label {{ font-size: 11px; color: #64748b; }}
        .param-value {{ font-size: 14px; font-weight: bold; color: #1e293b; }}
        .tracker {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 15px; margin-top: 15px; }}
        .footer {{ text-align: center; padding: 20px; color: #64748b; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇸🇦 TASI Trend-Based Leverage Strategy</h1>
            <p>Daily Report - {scanner.scan_time.strftime('%A, %B %d, %Y')}</p>
            <div class="regime-badge">{regime_emoji} {regime} MARKET - {leverage}x LEVERAGE</div>
        </div>
        
        <div class="section">
            <h2>📊 Strategy Overview</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{leverage}x</div>
                    <div class="metric-label">Current Leverage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{scanner.capital * leverage:,.0f}</div>
                    <div class="metric-label">Target Exposure (SAR)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(scanner.positions)}</div>
                    <div class="metric-label">Active Positions</div>
                </div>
            </div>
            
            <div class="warning">
                <strong>⚠️ Regime Change Alert:</strong> 
                If market proxy price crosses <strong>{market_ma:.2f} SAR</strong>, 
                regime will change to {"BEAR (0.5x leverage)" if regime == "BULL" else "BULL (1.5x leverage)"}.
                Current price: <strong>{market_price:.2f} SAR</strong> ({(market_price/market_ma-1)*100:+.1f}% from trigger).
            </div>
        </div>
        
        <div class="section">
            <h2>💼 Position Allocation</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Name</th>
                    <th>Price</th>
                    <th>50-MA</th>
                    <th>vs MA</th>
                    <th>Weight</th>
                    <th>Shares</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
"""
        
        for p in scanner.positions:
            status_class = 'bull' if p.stock_regime == 'BULL' else 'bear'
            status_emoji = '🟢' if p.stock_regime == 'BULL' else '🔴'
            html += f"""
                <tr>
                    <td><strong>{p.ticker}</strong></td>
                    <td>{p.name}</td>
                    <td>{p.current_price:.2f}</td>
                    <td>{p.ma_50:.2f}</td>
                    <td class="{status_class}">{p.price_vs_ma_pct:+.1f}%</td>
                    <td>{p.target_weight_pct:.1f}%</td>
                    <td>{p.target_shares:,}</td>
                    <td>{p.target_value:,.0f}</td>
                    <td class="{status_class}">{status_emoji} {p.stock_regime}</td>
                </tr>
"""
        
        total_value = sum(p.target_value for p in scanner.positions)
        total_weight = sum(p.target_weight_pct for p in scanner.positions)
        
        html += f"""
                <tr style="background: #f1f5f9; font-weight: bold;">
                    <td colspan="5">TOTAL</td>
                    <td>{total_weight:.1f}%</td>
                    <td></td>
                    <td>{total_value:,.0f}</td>
                    <td></td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📋 Top 5 Position Details</h2>
"""
        
        for p in scanner.positions[:5]:
            status_emoji = '🟢' if p.stock_regime == 'BULL' else '🔴'
            html += f"""
            <div class="trade-card">
                <h3>{p.ticker} - {p.name} {status_emoji}</h3>
                <div class="trade-params">
                    <div class="param">
                        <div class="param-label">Entry Price</div>
                        <div class="param-value">{p.current_price:.2f} SAR</div>
                    </div>
                    <div class="param">
                        <div class="param-label">Target Shares</div>
                        <div class="param-value">{p.target_shares:,}</div>
                    </div>
                    <div class="param">
                        <div class="param-label">Position Value</div>
                        <div class="param-value">{p.target_value:,.0f} SAR</div>
                    </div>
                    <div class="param">
                        <div class="param-label">Portfolio Weight</div>
                        <div class="param-value">{p.target_weight_pct:.1f}%</div>
                    </div>
                    <div class="param">
                        <div class="param-label">50-Day MA</div>
                        <div class="param-value">{p.ma_50:.2f} SAR</div>
                    </div>
                    <div class="param">
                        <div class="param-label">Regime Change Price</div>
                        <div class="param-value">{p.regime_change_price:.2f} SAR</div>
                    </div>
                </div>
            </div>
"""
        
        html += """
        </div>
        
        <div class="section">
            <h2>📈 Performance Tracking</h2>
"""
        
        if scanner.tracker and scanner.tracker.regime_history:
            bull_days = sum(r.days for r in scanner.tracker.regime_history if r.regime == 'BULL')
            bear_days = sum(r.days for r in scanner.tracker.regime_history if r.regime == 'BEAR')
            total_days = scanner.tracker.total_days
            
            html += f"""
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{total_days}</div>
                    <div class="metric-label">Days Tracked</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{bull_days}</div>
                    <div class="metric-label">Bull Days ({bull_days/max(1,total_days)*100:.0f}%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(scanner.tracker.regime_history)}</div>
                    <div class="metric-label">Regime Changes</div>
                </div>
            </div>
            
            <div class="tracker">
                <strong>📊 Regime History (Last 5):</strong>
                <table style="margin-top: 10px;">
                    <tr>
                        <th>Start</th>
                        <th>End</th>
                        <th>Regime</th>
                        <th>Leverage</th>
                        <th>Days</th>
                        <th>Status</th>
                    </tr>
"""
            for r in scanner.tracker.regime_history[-5:]:
                end_str = r.end_date if r.end_date else "Ongoing"
                status_class = 'bull' if r.regime == 'BULL' else 'bear'
                html += f"""
                    <tr>
                        <td>{r.start_date}</td>
                        <td>{end_str}</td>
                        <td class="{status_class}">{r.regime}</td>
                        <td>{r.leverage}x</td>
                        <td>{r.days}</td>
                        <td>{r.status}</td>
                    </tr>
"""
            html += """
                </table>
            </div>
"""
        else:
            html += """
            <p>Tracking just started. Run daily to build performance history.</p>
"""
        
        html += f"""
        </div>
        
        <div class="section">
            <h2>📊 Expected Performance (Backtest 2022-2026)</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Buy & Hold</th>
                    <th>This Strategy</th>
                    <th>Excess</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>+33.8%</td>
                    <td class="bull">+141.7%</td>
                    <td class="bull">+107.9%</td>
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td>+7.4%</td>
                    <td class="bull">+24.3%</td>
                    <td class="bull">+16.8%</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>0.56</td>
                    <td class="bull">1.61</td>
                    <td class="bull">+1.05</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>19.3%</td>
                    <td class="bull">11.2%</td>
                    <td class="bull">Better</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>TASI Trend-Based Leverage Strategy | Generated {scanner.scan_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="font-size: 10px; margin-top: 10px;">
                This strategy is for informational purposes only. Past performance does not guarantee future results.
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def send_report(self, scanner: CompleteScanner, regime: str, leverage: float,
                   market_price: float, market_ma: float) -> bool:
        """Send email report."""
        
        if not all([self.email_user, self.email_pass, self.recipient]):
            print("Email credentials not configured")
            return False
        
        try:
            html = self.generate_html(scanner, regime, leverage, market_price, market_ma)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🇸🇦 TASI Strategy: {regime} Market - {leverage}x Leverage | {scanner.scan_time.strftime('%Y-%m-%d')}"
            msg['From'] = self.email_user
            msg['To'] = self.recipient
            
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                server.sendmail(self.email_user, self.recipient, msg.as_string())
            
            print(f"Email sent to {self.recipient}")
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False


def main():
    """Run scanner and send email."""
    from dotenv import load_dotenv
    load_dotenv()
    
    scanner = CompleteScanner(capital=1_000_000)
    report = scanner.run()
    
    # Print console report
    print(report)
    
    # Send email
    emailer = CompleteEmailer()
    
    if scanner.tracker:
        emailer.send_report(
            scanner,
            scanner.tracker.current_regime,
            scanner.tracker.current_leverage,
            scanner.stock_data['1180.SR']['close'].iloc[-1],
            scanner.stock_data['1180.SR']['close'].rolling(50).mean().iloc[-1]
        )


if __name__ == "__main__":
    main()
