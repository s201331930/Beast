"""
PRODUCTION CONFIGURATION
========================
All strategy parameters in one place for easy adjustment.
"""

# =============================================================================
# STRATEGY PARAMETERS (Scientifically Optimized)
# =============================================================================

STRATEGY = {
    # Regime Detection
    'regime_ma_period': 50,           # 50-day MA for regime detection
    'regime_vol_threshold': 0.25,     # High volatility threshold (annualized)
    
    # Leverage by Regime
    'leverage_bull': 1.5,             # Leverage in bull market
    'leverage_neutral': 1.0,          # Leverage in neutral market
    'leverage_bear': 0.5,             # Leverage in bear market
    'leverage_high_vol_multiplier': 0.7,  # Reduce leverage in high vol
    
    # Position Sizing
    'max_position_pct': 0.10,         # Max 10% per stock
    'min_position_pct': 0.03,         # Min 3% per stock
    'max_positions': 15,              # Maximum concurrent positions
    
    # Entry Signals
    'rsi_period': 14,
    'rsi_oversold': 35,               # RSI threshold for oversold
    'rsi_overbought': 70,             # RSI threshold for overbought
    'bb_period': 20,
    'bb_std': 2,
    'volume_ratio_threshold': 1.5,    # Volume must be 1.5x average
    'momentum_period': 20,
    'momentum_threshold': 0.05,       # 5% momentum threshold
    
    # Exit Rules
    'take_profit_pct': 0.25,          # 25% take profit
    'trailing_stop_pct': 0.07,        # 7% trailing stop
    'trailing_activation_pct': 0.05,  # Activate trailing after 5% gain
    'max_holding_days': 180,          # Maximum holding period
    
    # Scoring Weights
    'score_weights': {
        'momentum': 0.20,
        'volatility': 0.15,
        'volume': 0.15,
        'rsi': 0.15,
        'trend': 0.20,
        'beta': 0.15,
    },
    'min_score': 60,                  # Minimum score to trade
}

# =============================================================================
# TASI STOCK UNIVERSE
# =============================================================================

TASI_STOCKS = {
    # Banks
    '1180.SR': {'name': 'Al Rajhi Bank', 'sector': 'Banks', 'priority': 1},
    '1010.SR': {'name': 'Riyad Bank', 'sector': 'Banks', 'priority': 1},
    '1050.SR': {'name': 'Banque Saudi Fransi', 'sector': 'Banks', 'priority': 2},
    '1060.SR': {'name': 'Saudi Awwal Bank', 'sector': 'Banks', 'priority': 2},
    '1080.SR': {'name': 'Arab National Bank', 'sector': 'Banks', 'priority': 2},
    '1120.SR': {'name': 'Al Jazira Bank', 'sector': 'Banks', 'priority': 3},
    '1140.SR': {'name': 'Bank Albilad', 'sector': 'Banks', 'priority': 2},
    '1150.SR': {'name': 'Alinma Bank', 'sector': 'Banks', 'priority': 1},
    
    # Telecom
    '7010.SR': {'name': 'STC', 'sector': 'Telecom', 'priority': 1},
    '7020.SR': {'name': 'Mobily', 'sector': 'Telecom', 'priority': 2},
    '7030.SR': {'name': 'Zain KSA', 'sector': 'Telecom', 'priority': 3},
    
    # Materials
    '2010.SR': {'name': 'SABIC', 'sector': 'Materials', 'priority': 1},
    '1211.SR': {'name': 'Maaden', 'sector': 'Materials', 'priority': 1},
    '2290.SR': {'name': 'Yanbu National Petro', 'sector': 'Materials', 'priority': 2},
    '2310.SR': {'name': 'SIIG', 'sector': 'Materials', 'priority': 2},
    '2330.SR': {'name': 'Advanced Petrochem', 'sector': 'Materials', 'priority': 2},
    '2350.SR': {'name': 'Saudi Kayan', 'sector': 'Materials', 'priority': 3},
    '1320.SR': {'name': 'Saudi Steel Pipe', 'sector': 'Materials', 'priority': 2},
    '1304.SR': {'name': 'Yamamah Steel', 'sector': 'Materials', 'priority': 3},
    '1321.SR': {'name': 'East Pipes', 'sector': 'Materials', 'priority': 3},
    
    # Cement
    '3020.SR': {'name': 'Yamama Cement', 'sector': 'Cement', 'priority': 2},
    '3030.SR': {'name': 'Saudi Cement', 'sector': 'Cement', 'priority': 2},
    '3040.SR': {'name': 'Qassim Cement', 'sector': 'Cement', 'priority': 3},
    '3050.SR': {'name': 'Southern Cement', 'sector': 'Cement', 'priority': 3},
    '3060.SR': {'name': 'Yanbu Cement', 'sector': 'Cement', 'priority': 3},
    
    # Real Estate
    '4300.SR': {'name': 'Dar Al Arkan', 'sector': 'Real Estate', 'priority': 2},
    '4310.SR': {'name': 'Emaar Economic City', 'sector': 'Real Estate', 'priority': 3},
    '4320.SR': {'name': 'Al Andalus', 'sector': 'Real Estate', 'priority': 3},
    '4250.SR': {'name': 'Jabal Omar', 'sector': 'Real Estate', 'priority': 2},
    
    # Retail
    '4190.SR': {'name': 'Jarir Marketing', 'sector': 'Retail', 'priority': 1},
    '4001.SR': {'name': 'Abdullah Al Othaim', 'sector': 'Retail', 'priority': 2},
    '4003.SR': {'name': 'Extra', 'sector': 'Retail', 'priority': 3},
    '4006.SR': {'name': 'Fawaz Al Hokair', 'sector': 'Retail', 'priority': 3},
    
    # Food & Beverages
    '2280.SR': {'name': 'Almarai', 'sector': 'Food', 'priority': 1},
    '2050.SR': {'name': 'Savola', 'sector': 'Food', 'priority': 2},
    '6010.SR': {'name': 'Nadec', 'sector': 'Food', 'priority': 3},
    '6020.SR': {'name': 'Sadafco', 'sector': 'Food', 'priority': 3},
    '2270.SR': {'name': 'Saudia Dairy', 'sector': 'Food', 'priority': 3},
    
    # Healthcare
    '4002.SR': {'name': 'Mouwasat', 'sector': 'Healthcare', 'priority': 2},
    '4004.SR': {'name': 'Dallah Healthcare', 'sector': 'Healthcare', 'priority': 2},
    '4005.SR': {'name': 'Care', 'sector': 'Healthcare', 'priority': 3},
    '4007.SR': {'name': 'Al Hammadi', 'sector': 'Healthcare', 'priority': 3},
    '4009.SR': {'name': 'Middle East Healthcare', 'sector': 'Healthcare', 'priority': 2},
    
    # Insurance
    '8210.SR': {'name': 'Bupa Arabia', 'sector': 'Insurance', 'priority': 1},
    '8010.SR': {'name': 'Tawuniya', 'sector': 'Insurance', 'priority': 2},
    '8200.SR': {'name': 'Saudi Re', 'sector': 'Insurance', 'priority': 3},
    
    # Energy
    '2222.SR': {'name': 'Saudi Aramco', 'sector': 'Energy', 'priority': 1},
    '2380.SR': {'name': 'Petro Rabigh', 'sector': 'Energy', 'priority': 2},
    '2030.SR': {'name': 'SARCO', 'sector': 'Energy', 'priority': 3},
    
    # Utilities
    '5110.SR': {'name': 'Saudi Electricity', 'sector': 'Utilities', 'priority': 2},
    '2082.SR': {'name': 'ACWA Power', 'sector': 'Utilities', 'priority': 1},
    '2083.SR': {'name': 'Marafiq', 'sector': 'Utilities', 'priority': 3},
    
    # Transportation
    '4030.SR': {'name': 'Bahri', 'sector': 'Transportation', 'priority': 2},
    '4031.SR': {'name': 'SAL', 'sector': 'Transportation', 'priority': 3},
    '4040.SR': {'name': 'Saudi Ground Services', 'sector': 'Transportation', 'priority': 3},
    '4261.SR': {'name': 'Mohammed Al-Mojil', 'sector': 'Transportation', 'priority': 3},
    
    # Capital Goods
    '1212.SR': {'name': 'Astra Industrial', 'sector': 'Industrial', 'priority': 2},
    '2060.SR': {'name': 'National Industrialization', 'sector': 'Industrial', 'priority': 2},
    '2170.SR': {'name': 'Alujain', 'sector': 'Industrial', 'priority': 3},
    '2240.SR': {'name': 'Zamil Industrial', 'sector': 'Industrial', 'priority': 3},
}

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

SYSTEM = {
    'data_lookback_days': 365,        # Days of historical data to fetch
    'min_data_days': 100,             # Minimum days required for analysis
    'cache_expiry_hours': 4,          # Cache data for 4 hours
    'output_dir': 'output/production',
    'log_level': 'INFO',
}

# =============================================================================
# EMAIL SETTINGS (for daily reports)
# =============================================================================

EMAIL = {
    'enabled': True,
    'recipient': 'n.aljudayi@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
}
