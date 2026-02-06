#!/usr/bin/env python3
"""Generate sample HTML output to file."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production.complete_scanner import CompleteScanner
from production.complete_emailer import CompleteEmailer


def main():
    scanner = CompleteScanner(capital=1_000_000)
    report = scanner.run()
    
    # Generate HTML
    emailer = CompleteEmailer()
    
    market_proxy = '1180.SR'
    market_price = scanner.stock_data[market_proxy]['close'].iloc[-1]
    market_ma = scanner.stock_data[market_proxy]['close'].rolling(50).mean().iloc[-1]
    
    html = emailer.generate_html(
        scanner,
        scanner.tracker.current_regime,
        scanner.tracker.current_leverage,
        market_price,
        market_ma
    )
    
    # Save HTML
    os.makedirs("output/production", exist_ok=True)
    with open("output/production/sample_email.html", 'w') as f:
        f.write(html)
    
    print("Sample HTML saved to: output/production/sample_email.html")
    print(f"\nSample console report:\n")
    print(report)


if __name__ == "__main__":
    main()
