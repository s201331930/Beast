# Stock Anomaly Prediction System

## A State-of-the-Art System for Detecting and Predicting Big Stock Moves

This comprehensive system combines mathematical, statistical, and physics-inspired models to detect anomalies that may precede significant stock price movements (rallies). Built with RKLB (Rocket Lab USA) as the primary case study, but applicable to any liquid stock.

## 🎯 Objectives

1. **Detect anomalies** in price, volume, and sentiment data before big moves
2. **Predict rallies** using multi-factor ensemble models
3. **Backtest rigorously** with walk-forward optimization
4. **Provide actionable signals** with confidence scores

## 🧮 Mathematical & Scientific Foundation

### Statistical Anomaly Detection
- **Z-Score Analysis**: Gaussian assumption, rolling standardization
- **Modified Z-Score**: Robust to outliers using MAD (Median Absolute Deviation)
- **Grubbs Test**: Single outlier detection with t-distribution critical values
- **Generalized ESD**: Multiple outlier detection
- **IQR Method**: Non-parametric outlier detection
- **Mahalanobis Distance**: Multivariate anomaly detection accounting for correlations
- **CUSUM**: Cumulative sum control chart for change detection
- **EWMA**: Exponentially weighted control charts
- **Kolmogorov-Smirnov**: Distribution shift detection

### Machine Learning Models
- **Isolation Forest**: Tree-based anomaly isolation (O(n log n))
- **One-Class SVM**: Kernel-based decision boundary
- **Local Outlier Factor (LOF)**: Density-based local anomaly detection
- **DBSCAN Clustering**: Noise point detection
- **Gaussian Mixture Models**: Probabilistic clustering
- **Autoencoders**: Neural network reconstruction error
- **LSTM Autoencoders**: Temporal pattern anomaly detection

### Physics-Inspired Cyclical Models
- **Fourier Transform**: Frequency domain decomposition for cycle detection
- **Hilbert Transform**: Instantaneous phase and amplitude extraction
- **Wavelet Transform**: Multi-resolution time-frequency analysis
- **Hurst Exponent**: Long-range dependence measurement (R/S analysis)
- **Ornstein-Uhlenbeck Process**: Mean reversion modeling with half-life
- **Detrended Fluctuation Analysis (DFA)**: Scaling exponent for correlations
- **Spectral Entropy**: Market complexity/predictability measurement

### Sentiment & Alternative Data
- **VADER Sentiment**: Social media text analysis
- **TextBlob**: Linguistic sentiment analysis
- **Custom Financial Lexicon**: Domain-specific sentiment words
- **Google Trends**: Search interest anomaly detection
- **Options Flow**: Put/Call ratio contrarian signals

### Market Context
- **VIX Analysis**: Volatility regime detection and contrarian signals
- **Beta Analysis**: Systematic risk measurement
- **Sector Rotation**: Space sector momentum analysis
- **Peer Correlation**: Decorrelation detection
- **Risk-On/Risk-Off Scoring**: Market regime classification

## 📊 Key Metrics & Formulas

### Hurst Exponent (H)
```
E[R(n)/S(n)] = C × n^H

H < 0.5: Mean reverting (anti-persistent)
H = 0.5: Random walk (Brownian motion)
H > 0.5: Trending (persistent)
```

### Ornstein-Uhlenbeck Half-Life
```
dx = θ(μ - x)dt + σdW
Half-life = ln(2) / θ
```

### Isolation Forest Anomaly Score
```
s(x, n) = 2^(-E[h(x)] / c(n))
```
Where h(x) is path length and c(n) is average path length in random tree.

### Sharpe Ratio
```
Sharpe = (R_p - R_f) / σ_p × √252
```

## 🏗️ Architecture

```
stock_anomaly_predictor/
├── config/
│   └── settings.py          # Configuration dataclasses
├── data/
│   └── collector.py         # Multi-source data collection
├── models/
│   ├── statistical_anomaly.py   # Statistical methods
│   ├── ml_anomaly.py            # ML-based detection
│   ├── cyclical_models.py       # Physics-inspired models
│   └── signal_aggregator.py     # Ensemble combination
├── analysis/
│   ├── sentiment.py         # Sentiment analysis
│   └── market_context.py    # Market correlation analysis
├── backtest/
│   └── backtester.py        # Comprehensive backtesting
├── visualization/
│   └── dashboard.py         # Charts and reports
├── main.py                  # Main pipeline
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
cd stock_anomaly_predictor
pip install -r requirements.txt
```

### Basic Usage

```python
from main import StockAnomalyPredictor

# Initialize predictor
predictor = StockAnomalyPredictor("RKLB")

# Run full analysis
results = predictor.run_full_pipeline(
    include_deep_learning=True,
    run_backtest=True,
    generate_reports=True
)

# Get current signal
signal = predictor.get_current_signal()
print(f"Composite Score: {signal['composite_score']}")
print(f"Alert Level: {signal['alert_level']}")
```

### Command Line

```bash
# Quick analysis
python main.py --ticker RKLB --quick

# Full analysis with backtesting
python main.py --ticker RKLB --full-analysis

# Custom ticker
python main.py --ticker TSLA --backtest
```

## 📈 Signal Interpretation

### Composite Score (0-100)
- **85-100**: Very High - Strong rally potential
- **70-85**: High - Favorable conditions
- **55-70**: Moderate - Mixed signals
- **40-55**: Low - Neutral/consolidation
- **0-40**: Very Low - Bearish conditions

### Key Signals

| Signal | Range | Interpretation |
|--------|-------|----------------|
| Rally Probability | 0-1 | Probability of >5% move in 5 days |
| Anomaly Intensity | 0-1 | How unusual current conditions are |
| Signal Confidence | 0-1 | Agreement between signal sources |
| Directional Bias | -1 to +1 | Bullish (+) vs Bearish (-) |

## 🔬 Backtesting Framework

### Features
- **Walk-Forward Optimization**: Rolling out-of-sample testing
- **Monte Carlo Simulation**: Robustness via trade shuffling
- **Transaction Costs**: Realistic commission and slippage
- **Risk Management**: Stop-loss, take-profit, trailing stops
- **Statistical Significance**: T-tests for signal validity

### Key Metrics Tracked
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, VaR, CVaR
- Win Rate, Profit Factor, Recovery Factor
- Alpha, Beta, Information Ratio

## 🔧 Configuration

All parameters are configurable in `config/settings.py`:

```python
@dataclass
class AnomalyConfig:
    zscore_threshold: float = 2.5
    volume_ratio_threshold: float = 2.5
    big_move_threshold: float = 0.05  # 5%
    
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
```

## 📊 Data Sources

| Source | Data Type | Notes |
|--------|-----------|-------|
| Yahoo Finance | Price, Volume, Options | Primary source |
| Google Trends | Search Interest | Requires pytrends |
| Twitter/X | Social Sentiment | Requires API key |
| News API | News Sentiment | Requires API key |
| FRED | Economic Data | Optional |

## 🎯 RKLB-Specific Insights

For Rocket Lab (RKLB), key signals often include:
- **Launch announcements**: Volume spikes before news
- **Contract wins**: Sentiment anomalies
- **Sector momentum**: Space ETF (ARKX) correlation
- **Small cap behavior**: High beta, momentum-driven

## ⚠️ Disclaimers

1. **Not Financial Advice**: This system is for educational and research purposes
2. **Past Performance**: Backtested results do not guarantee future returns
3. **Risk**: All trading involves substantial risk of loss
4. **Data Quality**: Results depend on data accuracy and availability

## 🔮 Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Options flow integration
- [ ] Earnings call transcript analysis
- [ ] SEC filing sentiment
- [ ] Order flow imbalance
- [ ] Cross-asset correlation networks
- [ ] Reinforcement learning position sizing

## 📚 References

### Academic Papers
- Isolation Forest: Liu, F.T., Ting, K.M., Zhou, Z.H. (2008)
- CUSUM: Page, E.S. (1954)
- Hurst Exponent: Hurst, H.E. (1951)
- LOF: Breunig, M.M., et al. (2000)

### Books
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Marcos López de Prado
- "Quantitative Trading" - Ernest Chan

## License

MIT License - Use at your own risk.

---

**Built with precision and pride. May your signals be strong and your drawdowns be small.** 🚀📈
