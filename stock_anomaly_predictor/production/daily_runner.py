#!/usr/bin/env python3
"""
DAILY RUNNER
============
Production script that:
1. Scans market for new signals
2. Updates existing tracked trades
3. Generates comprehensive report
4. Sends email with results

Run this daily after market close.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    os.system("pip install yfinance --quiet")
    import yfinance as yf

from production.config import STRATEGY, TASI_STOCKS, SYSTEM, EMAIL
from production.scanner import TASIScanner, StockAnalysis
from production.tracker import TradeTracker, TrackedTrade


class DailyRunner:
    """
    Orchestrates daily scanning and tracking.
    """
    
    def __init__(self):
        self.scanner = TASIScanner(quick_mode=False)
        self.tracker = TradeTracker()
        self.run_time = datetime.now()
        self.new_signals: List[StockAnalysis] = []
        self.updated_trades: List[TrackedTrade] = []
    
    def run(self) -> Dict:
        """Run full daily process."""
        print("=" * 80)
        print(f"DAILY RUNNER - {self.run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Run scanner
        print("\n[STEP 1] SCANNING MARKET...")
        self.scanner.run_scan()
        self.new_signals = self.scanner.get_buy_signals()
        
        # Step 2: Get current prices for tracked trades
        print("\n[STEP 2] UPDATING TRACKED TRADES...")
        current_prices = self._get_current_prices()
        
        # Update existing trades
        active_before = len(self.tracker.get_active_trades())
        self.updated_trades = self.tracker.update_all_trades(current_prices)
        active_after = len(self.tracker.get_active_trades())
        closed_today = active_before - active_after
        
        print(f"  Updated {len(self.updated_trades)} active trades")
        print(f"  Closed today: {closed_today}")
        
        # Step 3: Add new signals to tracker (if not already tracked)
        print("\n[STEP 3] PROCESSING NEW SIGNALS...")
        new_added = 0
        tracked_tickers = {t.ticker for t in self.tracker.get_active_trades()}
        
        for signal in self.new_signals:
            if signal.ticker not in tracked_tickers:
                self.tracker.add_trade({
                    'ticker': signal.ticker,
                    'name': signal.name,
                    'sector': signal.sector,
                    'signal': signal.signal,
                    'signal_strength': signal.signal_strength,
                    'regime': signal.regime,
                    'leverage': signal.leverage,
                    'score': signal.score,
                    'entry_price': signal.entry_price,
                    'take_profit': signal.take_profit,
                    'trailing_stop': signal.trailing_stop,
                    'current_price': signal.current_price,
                    'position_size_pct': signal.position_size_pct
                })
                new_added += 1
                print(f"  + Added: {signal.ticker} - {signal.name}")
        
        print(f"  New trades added: {new_added}")
        
        # Step 4: Generate summary
        summary = self.tracker.get_summary()
        
        print(f"\n[SUMMARY]")
        print(f"  Active Trades: {summary['active_trades']}")
        print(f"  Closed Trades: {summary['closed_trades']}")
        print(f"  Win Rate: {summary['win_rate']}%")
        print(f"  Avg Return: {summary['avg_return']:+.2f}%")
        
        return {
            'run_time': self.run_time.isoformat(),
            'market_regime': self.scanner.market_regime,
            'new_signals': len(self.new_signals),
            'new_added': new_added,
            'trades_updated': len(self.updated_trades),
            'trades_closed_today': closed_today,
            'summary': summary
        }
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all tracked tickers."""
        prices = {}
        tracked_tickers = {t.ticker for t in self.tracker.get_active_trades()}
        
        for ticker in tracked_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if len(hist) > 0:
                    prices[ticker] = hist['Close'].iloc[-1]
            except:
                pass
        
        return prices
    
    def generate_full_report(self) -> str:
        """Generate comprehensive daily report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("🇸🇦 TASI DAILY TRADING REPORT")
        lines.append(f"   Date: {self.run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        # Market Regime
        if self.scanner.market_regime:
            regime = self.scanner.market_regime
            lines.append(f"""
┌─────────────────────────────────────────────────────────────────┐
│  MARKET REGIME: {regime.regime:<20}                          │
│  Leverage Setting: {regime.leverage}x                                      │
│  Market Volatility: {regime.volatility*100:.1f}%                                   │
│  Trend Strength: {regime.trend_strength*100:+.1f}%                                   │
└─────────────────────────────────────────────────────────────────┘
""")
        
        # New Signals
        lines.append("=" * 80)
        lines.append(f"📈 NEW BUY SIGNALS ({len(self.new_signals)} stocks)")
        lines.append("=" * 80)
        
        if self.new_signals:
            lines.append(f"\n{'Ticker':<10} {'Name':<20} {'Score':>8} {'Entry':>10} {'T/P':>10} {'Stop':>10} {'Size':>8}")
            lines.append("-" * 90)
            
            for s in sorted(self.new_signals, key=lambda x: x.score, reverse=True):
                lines.append(f"{s.ticker:<10} {s.name[:20]:<20} {s.score:>8.1f} "
                           f"{s.entry_price:>10.2f} {s.take_profit:>10.2f} "
                           f"{s.trailing_stop:>10.2f} {s.position_size_pct:>7.1f}%")
            
            # Detailed parameters for each signal
            lines.append("\n" + "-" * 80)
            lines.append("TRADE DETAILS:")
            lines.append("-" * 80)
            
            for s in sorted(self.new_signals, key=lambda x: x.score, reverse=True):
                lines.append(f"""
{s.ticker} - {s.name}
  Regime: {s.regime} | Leverage: {s.leverage}x | Score: {s.score}/100
  Signal: {s.signal} (Strength: {s.signal_strength:.0%})
  ─────────────────────────────────────────────────────
  Entry Price:      {s.entry_price:.2f} SAR
  Take Profit:      {s.take_profit:.2f} SAR (+{STRATEGY['take_profit_pct']*100:.0f}%)
  Trailing Stop:    {s.trailing_stop:.2f} SAR (-{STRATEGY['trailing_stop_pct']*100:.0f}%)
  Position Size:    {s.position_size_pct:.1f}% of capital
  RSI: {s.details['rsi']:.1f} | Momentum: {s.details['momentum']:.1f}% | Vol: {s.details['volatility']:.1f}%
""")
        else:
            lines.append("\n  No new buy signals today.\n")
        
        # Active Trades Tracking
        active_trades = self.tracker.get_active_trades()
        lines.append("=" * 80)
        lines.append(f"📊 ACTIVE TRADES TRACKING ({len(active_trades)} positions)")
        lines.append("=" * 80)
        
        if active_trades:
            # Separate winners and losers
            winners = [t for t in active_trades if t.current_return_pct > 0]
            losers = [t for t in active_trades if t.current_return_pct <= 0]
            
            if winners:
                lines.append(f"\n  🟢 WINNING POSITIONS ({len(winners)}):")
                lines.append(f"  {'Ticker':<10} {'Entry':>10} {'Current':>10} {'Return':>10} {'Days':>6} {'T/P':>10} {'Stop':>10}")
                lines.append("  " + "-" * 80)
                for t in sorted(winners, key=lambda x: x.current_return_pct, reverse=True):
                    lines.append(f"  {t.ticker:<10} {t.entry_price:>10.2f} {t.current_price:>10.2f} "
                               f"{t.current_return_pct:>+9.2f}% {t.days_held:>6} "
                               f"{t.take_profit:>10.2f} {t.current_stop:>10.2f}")
            
            if losers:
                lines.append(f"\n  🔴 LOSING POSITIONS ({len(losers)}):")
                lines.append(f"  {'Ticker':<10} {'Entry':>10} {'Current':>10} {'Return':>10} {'Days':>6} {'T/P':>10} {'Stop':>10}")
                lines.append("  " + "-" * 80)
                for t in sorted(losers, key=lambda x: x.current_return_pct):
                    lines.append(f"  {t.ticker:<10} {t.entry_price:>10.2f} {t.current_price:>10.2f} "
                               f"{t.current_return_pct:>+9.2f}% {t.days_held:>6} "
                               f"{t.take_profit:>10.2f} {t.current_stop:>10.2f}")
            
            # Summary stats
            total_return = sum(t.current_return_pct for t in active_trades)
            avg_return = total_return / len(active_trades)
            total_exposure = sum(t.position_size_pct for t in active_trades)
            
            lines.append(f"""
  ─────────────────────────────────────────────────────
  Portfolio Stats:
    Total Positions: {len(active_trades)}
    Winning: {len(winners)} | Losing: {len(losers)}
    Avg Return: {avg_return:+.2f}%
    Total Exposure: {total_exposure:.1f}%
""")
        else:
            lines.append("\n  No active trades being tracked.\n")
        
        # Recently Closed Trades
        recent_closed = self.tracker.get_closed_trades(days=7)
        lines.append("=" * 80)
        lines.append(f"📋 RECENTLY CLOSED TRADES (Last 7 days)")
        lines.append("=" * 80)
        
        if recent_closed:
            lines.append(f"\n  {'Ticker':<10} {'Entry':>10} {'Exit':>10} {'Return':>10} {'Days':>6} {'Reason':<15}")
            lines.append("  " + "-" * 75)
            for t in sorted(recent_closed, key=lambda x: x.exit_date, reverse=True):
                emoji = "🟢" if t.final_return_pct > 0 else "🔴"
                lines.append(f"  {emoji} {t.ticker:<8} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
                           f"{t.final_return_pct:>+9.2f}% {t.days_held:>6} {t.exit_reason:<15}")
        else:
            lines.append("\n  No trades closed in the last 7 days.\n")
        
        # Performance Summary
        summary = self.tracker.get_summary()
        lines.append("=" * 80)
        lines.append("📈 OVERALL PERFORMANCE")
        lines.append("=" * 80)
        
        lines.append(f"""
  Total Closed Trades:  {summary['closed_trades']}
  Win Rate:             {summary['win_rate']:.1f}%
  Average Return:       {summary['avg_return']:+.2f}%
  Average Win:          {summary.get('avg_win', 0):+.2f}%
  Average Loss:         {summary.get('avg_loss', 0):+.2f}%
  Total Return:         {summary['total_return']:+.2f}%
""")
        
        # Strategy parameters reminder
        lines.append("=" * 80)
        lines.append("⚙️ STRATEGY PARAMETERS")
        lines.append("=" * 80)
        lines.append(f"""
  Regime Detection:     50-day Moving Average
  Bull Leverage:        {STRATEGY['leverage_bull']}x
  Bear Leverage:        {STRATEGY['leverage_bear']}x
  Take Profit:          {STRATEGY['take_profit_pct']*100:.0f}%
  Trailing Stop:        {STRATEGY['trailing_stop_pct']*100:.0f}% (activates at +{STRATEGY['trailing_activation_pct']*100:.0f}%)
  Max Holding Days:     {STRATEGY['max_holding_days']}
  Min Score to Trade:   {STRATEGY['min_score']}
""")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, output_dir: str = None) -> str:
        """Save report to file."""
        if output_dir is None:
            output_dir = SYSTEM['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.generate_full_report()
        timestamp = self.run_time.strftime('%Y%m%d')
        
        filepath = f"{output_dir}/daily_report_{timestamp}.txt"
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save as latest
        with open(f"{output_dir}/latest_daily_report.txt", 'w') as f:
            f.write(report)
        
        return filepath
    
    def send_email(self) -> bool:
        """Send report via email."""
        if not EMAIL['enabled']:
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            username = os.environ.get('SMTP_USERNAME', '')
            password = os.environ.get('SMTP_PASSWORD', '')
            
            if not username or not password:
                print("SMTP credentials not found")
                return False
            
            report = self.generate_full_report()
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🇸🇦 TASI Daily Report - {self.run_time.strftime('%Y-%m-%d')} | {len(self.new_signals)} Signals | {len(self.tracker.get_active_trades())} Active"
            msg['From'] = username
            msg['To'] = EMAIL['recipient']
            
            # Plain text
            msg.attach(MIMEText(report, 'plain'))
            
            # HTML version
            html = self._generate_html_report()
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(EMAIL['smtp_server'], EMAIL['smtp_port']) as server:
                server.starttls()
                server.login(username, password)
                server.sendmail(username, EMAIL['recipient'], msg.as_string())
            
            return True
            
        except Exception as e:
            print(f"Email error: {e}")
            return False
    
    def _generate_html_report(self) -> str:
        """Generate HTML email report."""
        regime = self.scanner.market_regime
        active_trades = self.tracker.get_active_trades()
        summary = self.tracker.get_summary()
        
        # Determine colors
        regime_color = "#28a745" if "BULL" in regime.regime else "#dc3545"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a5f2a; margin-bottom: 5px; }}
        h2 {{ color: #333; border-bottom: 2px solid #1a5f2a; padding-bottom: 8px; margin-top: 30px; }}
        .regime-box {{ background: {regime_color}; color: white; padding: 15px 20px; border-radius: 8px; margin: 20px 0; }}
        .regime-box h3 {{ margin: 0 0 10px 0; }}
        .signal-card {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .signal-header {{ font-size: 18px; font-weight: bold; color: #2e7d32; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #f0f0f0; padding: 12px 10px; text-align: left; border-bottom: 2px solid #ddd; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .win {{ color: #28a745; font-weight: bold; }}
        .loss {{ color: #dc3545; font-weight: bold; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1a5f2a; }}
        .stat-label {{ color: #666; font-size: 12px; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🇸🇦 TASI Daily Report</h1>
        <p style="color: #666; margin-top: 0;">{self.run_time.strftime('%A, %B %d, %Y')}</p>
        
        <div class="regime-box">
            <h3>Market Regime: {regime.regime}</h3>
            <p style="margin: 5px 0;">Leverage: {regime.leverage}x | Volatility: {regime.volatility*100:.1f}% | Trend: {regime.trend_strength*100:+.1f}%</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{len(self.new_signals)}</div>
                <div class="stat-label">New Signals</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(active_trades)}</div>
                <div class="stat-label">Active Trades</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{summary['win_rate']:.0f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
        </div>
"""
        
        # New Signals
        if self.new_signals:
            html += "<h2>📈 New Buy Signals</h2>"
            for s in sorted(self.new_signals, key=lambda x: x.score, reverse=True):
                html += f"""
                <div class="signal-card">
                    <div class="signal-header">{s.ticker} - {s.name}</div>
                    <table>
                        <tr><td>Score</td><td><strong>{s.score:.1f}/100</strong></td></tr>
                        <tr><td>Signal</td><td>{s.signal} ({s.signal_strength:.0%})</td></tr>
                        <tr><td>Entry Price</td><td><strong>{s.entry_price:.2f} SAR</strong></td></tr>
                        <tr><td>Take Profit</td><td>{s.take_profit:.2f} SAR (+{STRATEGY['take_profit_pct']*100:.0f}%)</td></tr>
                        <tr><td>Trailing Stop</td><td>{s.trailing_stop:.2f} SAR (-{STRATEGY['trailing_stop_pct']*100:.0f}%)</td></tr>
                        <tr><td>Position Size</td><td><strong>{s.position_size_pct:.1f}%</strong></td></tr>
                    </table>
                </div>
                """
        else:
            html += "<h2>📈 New Buy Signals</h2><p>No new signals today.</p>"
        
        # Active Trades
        html += f"<h2>📊 Active Trades ({len(active_trades)})</h2>"
        if active_trades:
            html += """<table>
                <tr><th>Ticker</th><th>Entry</th><th>Current</th><th>Return</th><th>Days</th><th>Status</th></tr>
            """
            for t in sorted(active_trades, key=lambda x: x.current_return_pct, reverse=True):
                ret_class = "win" if t.current_return_pct > 0 else "loss"
                html += f"""
                <tr>
                    <td><strong>{t.ticker}</strong></td>
                    <td>{t.entry_price:.2f}</td>
                    <td>{t.current_price:.2f}</td>
                    <td class="{ret_class}">{t.current_return_pct:+.2f}%</td>
                    <td>{t.days_held}</td>
                    <td>{'🟢' if t.current_return_pct > 0 else '🔴'}</td>
                </tr>
                """
            html += "</table>"
        else:
            html += "<p>No active trades.</p>"
        
        # Performance
        html += f"""
        <h2>📈 Performance Summary</h2>
        <table>
            <tr><td>Total Closed Trades</td><td><strong>{summary['closed_trades']}</strong></td></tr>
            <tr><td>Win Rate</td><td><strong>{summary['win_rate']:.1f}%</strong></td></tr>
            <tr><td>Average Return</td><td class="{'win' if summary['avg_return'] > 0 else 'loss'}">{summary['avg_return']:+.2f}%</td></tr>
            <tr><td>Total Return</td><td class="{'win' if summary['total_return'] > 0 else 'loss'}">{summary['total_return']:+.2f}%</td></tr>
        </table>
        
        <div class="footer">
            <p>Strategy: Regime-Adaptive Leverage | Take Profit: {STRATEGY['take_profit_pct']*100:.0f}% | Trailing Stop: {STRATEGY['trailing_stop_pct']*100:.0f}%</p>
            <p>Generated by TASI Scanner System</p>
        </div>
    </div>
</body>
</html>
"""
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='TASI Daily Runner')
    parser.add_argument('--email', action='store_true', help='Send email report')
    parser.add_argument('--no-save', action='store_true', help='Do not save report to file')
    args = parser.parse_args()
    
    runner = DailyRunner()
    
    # Run
    result = runner.run()
    
    # Generate and print report
    report = runner.generate_full_report()
    print("\n" + report)
    
    # Save
    if not args.no_save:
        filepath = runner.save_report()
        print(f"\nReport saved to: {filepath}")
    
    # Email
    if args.email:
        print("\nSending email...")
        if runner.send_email():
            print("Email sent successfully!")
        else:
            print("Failed to send email")


if __name__ == "__main__":
    main()
