#!/usr/bin/env python3
"""
Email Sender for TASI Daily Scanner
Sends the latest report via email

Works with:
- Local .env file
- GitHub Actions secrets (SMTP_USERNAME, SMTP_PASSWORD)
"""

import os
import sys
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import glob

# Load environment variables from .env file (if exists)
def load_env():
    """Load from .env file if available, otherwise use existing env vars"""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment (GitHub Actions sets secrets)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()
        print(f"✓ Loaded configuration from .env")
        return True
    elif os.environ.get('SMTP_USERNAME') and os.environ.get('SMTP_PASSWORD'):
        print(f"✓ Using environment variables (GitHub Actions)")
        return True
    else:
        print(f"✗ No credentials found")
        return False

def send_test_email():
    """Send test email with yesterday's report"""
    
    print("=" * 60)
    print("TASI Daily Scanner - Email Test")
    print("=" * 60)
    
    # Load config
    if not load_env():
        return False
    
    # Email settings
    sender = os.environ.get('SMTP_USERNAME')
    password = os.environ.get('SMTP_PASSWORD')
    recipient = 'n.aljudayi@gmail.com'
    
    if not sender or not password:
        print("✗ Email credentials not configured")
        return False
    
    print(f"✓ Sender: {sender}")
    print(f"✓ Recipient: {recipient}")
    
    # Find the latest report
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_pattern = os.path.join(script_dir, 'output/daily_scans/tasi_report_*.html')
    json_pattern = os.path.join(script_dir, 'output/daily_scans/tasi_scan_*.json')
    
    report_files = sorted(glob.glob(report_pattern), reverse=True)
    json_files = sorted(glob.glob(json_pattern), reverse=True)
    
    if not report_files:
        print(f"✗ No report files found in output/daily_scans/")
        return False
    
    report_file = report_files[0]  # Latest report
    json_file = json_files[0] if json_files else None
    
    print(f"✓ Loading report: {os.path.basename(report_file)}")
    
    with open(report_file, 'r') as f:
        html_content = f.read()
    
    scan_data = {'stocks_scanned': 0, 'stocks_qualified': 0, 'signals_generated': 0, 'active_signals': []}
    if json_file and os.path.exists(json_file):
        with open(json_file, 'r') as f:
            scan_data = json.load(f)
    
    # Extract date from report filename
    report_date = os.path.basename(report_file).replace('tasi_report_', '').replace('.html', '')
    if len(report_date) == 8:  # YYYYMMDD format
        formatted_date = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:8]}"
    else:
        formatted_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create email
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🇸🇦 TASI Daily Signals - {formatted_date} - {scan_data.get('signals_generated', 0)} Active Signals"
    msg['From'] = sender
    msg['To'] = recipient
    
    # Plain text version
    signals = scan_data.get('active_signals', [])
    
    # Get the most recent signal date
    if signals:
        latest_date = max(s['date'] for s in signals)
        latest_signals = [s for s in signals if s['date'] == latest_date]
    else:
        latest_date = formatted_date
        latest_signals = []
    
    text_content = f"""
TASI Daily Signal Report - {formatted_date}
=====================================

Scan Summary:
- Stocks Scanned: {scan_data.get('stocks_scanned', 'N/A')}
- Stocks Qualified: {scan_data.get('stocks_qualified', 'N/A')}
- Active Signals: {scan_data.get('signals_generated', 0)}

Latest Signals ({latest_date}):
"""
    for s in sorted(latest_signals, key=lambda x: -x.get('rally_probability', 0))[:10]:
        text_content += f"  • {s['ticker']}: {s.get('company', '')[:30]} - {s.get('rally_probability', 0):.0f}% probability @ {s.get('current_price', 0):.2f} SAR\n"
    
    if not latest_signals:
        text_content += "  (No new signals)\n"
    
    text_content += """

View the full HTML report for detailed analysis and charts.

---
TASI Anomaly Prediction System
Sent automatically at 7:00 PM KSA (Sunday-Thursday)
"""
    
    msg.attach(MIMEText(text_content, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))
    
    # Send email
    print("\nConnecting to Gmail SMTP server...")
    
    try:
        context = ssl.create_default_context()
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.set_debuglevel(0)  # Set to 1 for debug output
            server.ehlo()
            print("✓ Connected to smtp.gmail.com:587")
            
            server.starttls(context=context)
            print("✓ TLS encryption enabled")
            
            server.login(sender, password)
            print("✓ Authentication successful")
            
            server.sendmail(sender, recipient, msg.as_string())
            print("✓ Email sent successfully!")
        
        print("\n" + "=" * 60)
        print("SUCCESS! Test email sent to:", recipient)
        print("=" * 60)
        print("\nPlease check your inbox (and spam folder) for the test email.")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n✗ Authentication failed: {e}")
        print("  - Make sure you're using an App Password (not your regular password)")
        print("  - Check that the App Password is correct")
        return False
        
    except smtplib.SMTPException as e:
        print(f"\n✗ SMTP error: {e}")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = send_test_email()
    sys.exit(0 if success else 1)
