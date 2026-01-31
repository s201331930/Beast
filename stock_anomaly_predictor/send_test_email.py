#!/usr/bin/env python3
"""
Test Email Sender for TASI Daily Scanner
Sends yesterday's report to verify email configuration
"""

import os
import sys
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

# Load environment variables from .env file
def load_env():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✓ Loaded configuration from .env")
    else:
        print(f"✗ .env file not found")
        return False
    return True

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
    
    # Load yesterday's report
    report_file = 'output/daily_scans/tasi_report_20260130.html'
    json_file = 'output/daily_scans/tasi_scan_20260130.json'
    
    if not os.path.exists(report_file):
        print(f"✗ Report file not found: {report_file}")
        return False
    
    print(f"✓ Loading report: {report_file}")
    
    with open(report_file, 'r') as f:
        html_content = f.read()
    
    with open(json_file, 'r') as f:
        scan_data = json.load(f)
    
    # Create email
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🇸🇦 TASI Daily Signals - Test Email - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = sender
    msg['To'] = recipient
    
    # Plain text version
    signals = scan_data.get('active_signals', [])
    today_signals = [s for s in signals if s['date'] == '2026-01-29']
    
    text_content = f"""
TASI Daily Signal Report - TEST EMAIL
=====================================

This is a test email to verify your TASI Daily Scanner configuration.

Scan Summary:
- Stocks Scanned: {scan_data['stocks_scanned']}
- Stocks Qualified: {scan_data['stocks_qualified']}
- Active Signals: {scan_data['signals_generated']}

NEW Signals (Jan 29, 2026):
"""
    for s in sorted(today_signals, key=lambda x: -x['rally_probability']):
        text_content += f"  • {s['ticker']}: {s['company'][:30]} - {s['rally_probability']:.0f}% probability @ {s['current_price']:.2f} SAR\n"
    
    text_content += """

If you received this email, your TASI Daily Scanner is configured correctly!
Daily reports will be sent at 7:00 PM KSA time (Sunday-Thursday).

---
TASI Anomaly Prediction System
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
