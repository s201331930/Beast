"""
Visualization Dashboard
=======================
Comprehensive visualization for anomaly detection results:
- Price charts with signals
- Anomaly heatmaps
- Performance metrics
- Interactive dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class AnomalyVisualizer:
    """
    Creates comprehensive visualizations for anomaly detection results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        self.figsize = figsize
        self.colors = {
            'price': '#2E86AB',
            'buy_signal': '#28A745',
            'sell_signal': '#DC3545',
            'anomaly': '#FFC107',
            'volume': '#6C757D',
            'positive': '#28A745',
            'negative': '#DC3545'
        }
        
    def plot_price_with_signals(self, data: pd.DataFrame, signals: pd.Series,
                                title: str = "Price Chart with Anomaly Signals",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot price chart with buy/sell signals overlaid.
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, 
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price chart
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], color=self.colors['price'], 
                linewidth=1.5, label='Close Price')
        
        # Add moving averages
        if 'close' in data.columns:
            ma20 = data['close'].rolling(20).mean()
            ma50 = data['close'].rolling(50).mean()
            ax1.plot(data.index, ma20, color='orange', linewidth=1, 
                    alpha=0.7, label='MA20')
            ax1.plot(data.index, ma50, color='purple', linewidth=1, 
                    alpha=0.7, label='MA50')
        
        # Plot signals
        buy_signals = signals[signals > 0]
        if len(buy_signals) > 0:
            buy_prices = data.loc[buy_signals.index, 'close']
            ax1.scatter(buy_signals.index, buy_prices, 
                       color=self.colors['buy_signal'], marker='^', 
                       s=100, label='Buy Signal', zorder=5)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Volume
        ax2 = axes[1]
        colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
                  for i in range(len(data))]
        ax2.bar(data.index, data['volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Volume')
        
        # Highlight volume anomalies
        if 'Volume_spike' in data.columns:
            vol_anomalies = data[data['Volume_spike'] == 1].index
            for date in vol_anomalies:
                ax2.axvline(x=date, color='orange', alpha=0.3, linestyle='--')
        
        # Ensemble score
        ax3 = axes[2]
        if 'ensemble_score' in data.columns:
            ax3.fill_between(data.index, 0, data['ensemble_score'], 
                           where=data['ensemble_score'] >= 0,
                           color=self.colors['positive'], alpha=0.5, label='Bullish')
            ax3.fill_between(data.index, 0, data['ensemble_score'], 
                           where=data['ensemble_score'] < 0,
                           color=self.colors['negative'], alpha=0.5, label='Bearish')
            ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Signal Threshold')
            ax3.axhline(y=0.7, color='darkgreen', linestyle='--', alpha=0.5, label='Strong Signal')
        ax3.set_ylabel('Ensemble Score')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def plot_anomaly_heatmap(self, data: pd.DataFrame,
                            title: str = "Anomaly Detection Heatmap",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of anomaly signals across all models.
        """
        # Select anomaly columns
        anomaly_cols = [c for c in data.columns if '_anomaly' in c.lower()]
        
        if not anomaly_cols:
            logger.warning("No anomaly columns found")
            return None
        
        # Limit to last 100 days for readability
        recent_data = data[anomaly_cols].tail(100)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        im = ax.imshow(recent_data.T.values, aspect='auto', cmap='RdYlGn_r',
                      interpolation='nearest')
        
        # Labels
        ax.set_yticks(range(len(anomaly_cols)))
        ax.set_yticklabels([c.replace('_anomaly', '') for c in anomaly_cols], fontsize=8)
        
        # X-axis dates
        date_labels = recent_data.index[::10].strftime('%Y-%m-%d')
        ax.set_xticks(range(0, len(recent_data), 10))
        ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Anomaly Detector')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Anomaly (1=Yes, 0=No)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, data: pd.DataFrame,
                             title: str = "Model Signal Comparison",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare signals from different model categories.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # Price
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], color=self.colors['price'], linewidth=1.5)
        ax1.set_ylabel('Price ($)')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Statistical signals
        ax2 = axes[1]
        if 'stat_signal' in data.columns:
            ax2.fill_between(data.index, 0, data['stat_signal'], alpha=0.7, 
                           color='blue', label='Statistical')
        if 'stat_anomaly_count' in data.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(data.index, data['stat_anomaly_count'], 
                         color='darkblue', linestyle='--', alpha=0.5)
            ax2_twin.set_ylabel('Anomaly Count', color='darkblue')
        ax2.set_ylabel('Stat Signal')
        ax2.legend(loc='upper left')
        
        # ML signals
        ax3 = axes[2]
        if 'ml_signal' in data.columns:
            ax3.fill_between(data.index, 0, data['ml_signal'], alpha=0.7, 
                           color='green', label='Machine Learning')
        ax3.set_ylabel('ML Signal')
        ax3.legend(loc='upper left')
        
        # Cyclical/Sentiment signals
        ax4 = axes[3]
        if 'cyclical_signal' in data.columns:
            ax4.plot(data.index, data['cyclical_signal'], 
                    color='purple', label='Cyclical', alpha=0.7)
        if 'sentiment_signal' in data.columns:
            ax4.plot(data.index, data['sentiment_signal'], 
                    color='orange', label='Sentiment', alpha=0.7)
        ax4.set_ylabel('Signal Score')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_backtest_results(self, equity_curve: pd.Series, 
                             benchmark: pd.Series,
                             drawdown: pd.Series,
                             trades: List,
                             title: str = "Backtest Performance",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive backtest results.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_curve.index, equity_curve, 
                color=self.colors['positive'], linewidth=2, label='Strategy')
        
        # Normalize benchmark to same starting value
        benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        ax1.plot(benchmark.index, benchmark_normalized, 
                color='gray', linewidth=1.5, alpha=0.7, label='Benchmark (Buy & Hold)')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown.index, 0, drawdown * 100, 
                        color=self.colors['negative'], alpha=0.7)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Trade distribution
        ax3 = fig.add_subplot(gs[2, 0])
        if trades:
            trade_returns = [t.pnl_percent * 100 for t in trades if t.pnl_percent]
            colors = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                     for r in trade_returns]
            ax3.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_xlabel('Trade #')
            ax3.set_ylabel('Return (%)')
            ax3.set_title('Individual Trade Returns')
        
        # Trade return histogram
        ax4 = fig.add_subplot(gs[2, 1])
        if trades:
            ax4.hist(trade_returns, bins=20, color=self.colors['price'], 
                    alpha=0.7, edgecolor='black')
            ax4.axvline(x=np.mean(trade_returns), color='red', linestyle='--',
                       label=f'Mean: {np.mean(trade_returns):.1f}%')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Return Distribution')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_signal_analysis(self, predictions: pd.DataFrame,
                            title: str = "Signal Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze signal quality and forward returns.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Signal distribution over time
        ax1 = axes[0, 0]
        signal_counts = predictions['signal'].resample('M').sum()
        ax1.bar(signal_counts.index, signal_counts.values, width=20, 
               color=self.colors['buy_signal'], alpha=0.7)
        ax1.set_title('Monthly Signal Count')
        ax1.set_ylabel('Number of Signals')
        
        # Forward returns by signal strength
        ax2 = axes[0, 1]
        if 'ensemble_score' in predictions.columns and 'forward_return' in predictions.columns:
            # Bin by ensemble score
            predictions['score_bin'] = pd.cut(predictions['ensemble_score'], 
                                             bins=[-1, 0, 0.3, 0.5, 0.7, 1],
                                             labels=['<0', '0-0.3', '0.3-0.5', '0.5-0.7', '>0.7'])
            score_returns = predictions.groupby('score_bin')['forward_return'].mean() * 100
            score_returns.plot(kind='bar', ax=ax2, color=self.colors['price'], alpha=0.7)
            ax2.set_title('Average Forward Return by Signal Score')
            ax2.set_ylabel('Forward Return (%)')
            ax2.set_xlabel('Ensemble Score Range')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        
        # Confusion matrix visualization
        ax3 = axes[1, 0]
        if all(c in predictions.columns for c in ['correct', 'false_positive', 'missed_rally']):
            categories = ['True Positive', 'False Positive', 'Missed Rally']
            values = [
                predictions['correct'].sum(),
                predictions['false_positive'].sum(),
                predictions['missed_rally'].sum()
            ]
            colors = [self.colors['positive'], self.colors['negative'], 'gray']
            ax3.bar(categories, values, color=colors, alpha=0.7)
            ax3.set_title('Signal Accuracy Breakdown')
            ax3.set_ylabel('Count')
        
        # Cumulative returns following signals
        ax4 = axes[1, 1]
        signal_days = predictions[predictions['signal'] > 0].index
        if len(signal_days) > 0 and 'forward_return' in predictions.columns:
            signal_returns = predictions.loc[signal_days, 'forward_return'].dropna()
            cumulative = (1 + signal_returns).cumprod()
            ax4.plot(range(len(cumulative)), cumulative.values, 
                    color=self.colors['positive'], linewidth=2)
            ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax4.set_title('Cumulative Returns Following Signals')
            ax4.set_xlabel('Signal #')
            ax4.set_ylabel('Cumulative Return (1 = Starting Value)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_full_report(data: pd.DataFrame, signals: pd.Series,
                      backtest_results, predictions: pd.DataFrame,
                      output_dir: str = "reports") -> None:
    """
    Create full visualization report with all charts.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = AnomalyVisualizer()
    
    # 1. Price chart with signals
    visualizer.plot_price_with_signals(
        data, signals,
        title="RKLB Anomaly Detection Signals",
        save_path=f"{output_dir}/price_signals.png"
    )
    
    # 2. Anomaly heatmap
    visualizer.plot_anomaly_heatmap(
        data,
        title="RKLB Anomaly Detection Heatmap",
        save_path=f"{output_dir}/anomaly_heatmap.png"
    )
    
    # 3. Model comparison
    visualizer.plot_model_comparison(
        data,
        title="RKLB Model Signal Comparison",
        save_path=f"{output_dir}/model_comparison.png"
    )
    
    # 4. Backtest results
    if backtest_results:
        visualizer.plot_backtest_results(
            backtest_results.equity_curve,
            data['close'],
            backtest_results.drawdown_curve,
            backtest_results.trades,
            title="RKLB Strategy Backtest Performance",
            save_path=f"{output_dir}/backtest_results.png"
        )
    
    # 5. Signal analysis
    if predictions is not None:
        visualizer.plot_signal_analysis(
            predictions,
            title="RKLB Signal Quality Analysis",
            save_path=f"{output_dir}/signal_analysis.png"
        )
    
    logger.info(f"Full report saved to {output_dir}/")


if __name__ == "__main__":
    import yfinance as yf
    
    # Test visualization
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    data['ensemble_score'] = np.random.rand(len(data)) - 0.3
    
    signals = pd.Series(0, index=data.index)
    signals.iloc[::50] = 1  # Sample signals
    
    visualizer = AnomalyVisualizer()
    visualizer.plot_price_with_signals(data, signals, save_path="test_chart.png")
    print("Test chart saved")
