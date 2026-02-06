#!/usr/bin/env python3
"""
TASI Market Scanner - Quick Run Script
======================================
Run this to scan the Saudi market and get trading recommendations.

Usage:
    python run_scanner.py           # Full scan
    python run_scanner.py --quick   # Quick scan (top stocks only)
"""

import sys
import os

# Add production module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production.scanner import main

if __name__ == "__main__":
    main()
