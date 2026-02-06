#!/usr/bin/env python3
"""
DAILY RUNNER - Trend-Based Leverage Strategy
=============================================
Run this daily after market close to get:
1. Current market regime (BULL/BEAR)
2. Recommended leverage (1.5x or 0.5x)
3. Target portfolio allocations
4. Rebalance instructions if needed

Strategy Performance (Backtest):
- Total Return: +141.7% (vs Buy & Hold +33.8%)
- Sharpe Ratio: 1.61 (vs 0.56)
- Max Drawdown: 11.2% (vs 19.3%)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production.leverage_strategy import LeverageStrategyScanner

def main():
    scanner = LeverageStrategyScanner()
    state = scanner.run_scan()
    
    report = scanner.generate_report()
    print(report)
    
    scanner.save_results()
    
    # Summary for quick reference
    regime = state.market_regime
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"  Regime:     {regime.regime}")
    print(f"  Leverage:   {regime.leverage}x")
    print(f"  Price/MA:   {regime.price_vs_ma:+.1f}%")
    print(f"  Rebalance:  {'YES - ' + state.rebalance_reason if state.rebalance_needed else 'No'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
