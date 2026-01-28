# Stock Move Anomaly Prediction Study: RKLB

## Overview
This project implements a multi-disciplinary approach to predicting significant stock moves (rallies) for Rocket Lab (RKLB), using Anomaly Detection, Cyclical Analysis, and Statistical Modeling.

## Methodology

We adopted a three-pronged persona approach:

### 1. The Physicist (Anomaly Detection)
**Concept**: Phase transitions in physical systems (like water boiling) are often preceded by anomalous fluctuations.
**Implementation**: 
- **Isolation Forest**: Detects outliers in multidimensional feature space (Volume, VIX, Oil, Price).
- **One-Class SVM**: Defines a boundary for "normal" market behavior and flags deviations.
- **Hypothesis**: Big moves occur when the market state is fundamentally "abnormal".

### 2. The Mathematician (Cyclical Analysis)
**Concept**: Markets exhibit wave-like properties due to business cycles and human psychology.
**Implementation**:
- **Fourier Transform (FFT)**: Decomposes the price series into constituent frequencies to identify dominant cycles.
- **Features**: `cycle_period` and `cycle_magnitude` are used as inputs to the model.

### 3. The Statistician (Mean Reversion & Probability)
**Concept**: Asset prices tend to revert to the mean, but momentum can persist.
**Implementation**:
- **Z-Scores**: Rolling statistical normalization of returns and volume.
- **Hurst Exponent Proxy**: Measuring the tendency of a time series to regress to its mean or cluster in a direction.
- **Random Forest Classifier**: A supervised learning model that synthesizes all technical, cyclical, and statistical features to assign a probability of a rally.

## Data Sources
- **Target**: RKLB (Rocket Lab)
- **Macro**: S&P 500 (`^GSPC`), VIX (`^VIX`), Crude Oil (`CL=F`)
- **Alternative (Simulated)**: 
  - `twitter_volume`: Correlated with volatility.
  - `sentiment_score`: Momentum-based sentiment.
  - `google_trends`: Interest spikes.

## Results (Backtest)

The model was backtested on out-of-sample data (chronological split).

- **Baseline Return (5-day hold)**: ~4.85% (Avg return of random entry)
- **Strategy Return (5-day hold)**: ~6.92% (Avg return when model signals buy)
- **Precision**: ~79% (When the model predicts a rally, it is correct 79% of the time).
- **Recall**: ~6% (The model is highly conservative, flagging only the highest confidence setups).

### Top Predictive Features
1. **Bollinger Band %B (`bb_pct`)**: Relative position of price within volatility bands.
2. **S&P 500 (`sp500`)**: General market correlation.
3. **Volatility (`volatility_30`)**: Magnitude of recent price changes.
4. **Volume Ratio**: Current volume vs 20-day average.

## Usage

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Analysis**: `python src/models.py`
   - This will fetch data, train models, run the backtest, and save `model_results.csv`.

## Future Improvements
- **Live Data Integration**: Replace simulated sentiment with real X/Twitter API and Google Trends API.
- **Deep Learning**: Implement LSTM-Autoencoders for more complex sequence anomaly detection.
- **Regime Switching**: Explicitly model market regimes (Bull/Bear/Chop) using Hidden Markov Models.
