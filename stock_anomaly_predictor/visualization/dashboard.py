"""
Visualization and Reporting Dashboard

Comprehensive visualization of:
- Price and signal charts
- Anomaly detection results
- Backtest performance
- Equity curves and drawdowns
- Signal distribution analysis
- Regime visualization
- Interactive dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class DashboardGenerator:
    """
    Generate comprehensive visualization dashboards.
    """
    
    def __init__(self, figsize: tuple = (16, 10)):
        """
        Initialize dashboard generator.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = {
            'price': '#2E86AB',
            'buy': '#28A745',
            'sell': '#DC3545',
            'signal': '#FFC107',
            'volume': '#6C757D',
            'anomaly': '#FF6B6B',
            'equity': '#4ECDC4',
            'drawdown': '#FF6B6B',
            'benchmark': '#95A5A6'
        }
    
    def plot_price_with_signals(self,
                                price_df: pd.DataFrame,
                                signals_df: pd.DataFrame,
                                signal_column: str = 'buy_signal',
                                save_path: str = None) -> plt.Figure:
        """
        Plot price chart with buy signals overlaid.
        
        Args:
            price_df: Price DataFrame
            signals_df: Signals DataFrame
            signal_column: Column with buy signals
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2),
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price chart
        ax1 = axes[0]
        ax1.plot(price_df.index, price_df['close'], color=self.colors['price'], 
                 linewidth=1.5, label='Price')
        
        # Add moving averages if available
        for ma_col in ['sma_20', 'sma_50', 'sma_200']:
            if ma_col in price_df.columns:
                ax1.plot(price_df.index, price_df[ma_col], 
                        linewidth=1, alpha=0.7, label=ma_col.upper())
        
        # Plot buy signals
        if signal_column in signals_df.columns:
            buy_dates = signals_df[signals_df[signal_column] == True].index
            buy_prices = price_df.loc[buy_dates, 'close']
            ax1.scatter(buy_dates, buy_prices, color=self.colors['buy'], 
                       marker='^', s=100, label='Buy Signal', zorder=5)
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'Price Chart with Signals', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2 = axes[1]
        if 'volume' in price_df.columns:
            colors = ['green' if price_df['close'].iloc[i] >= price_df['close'].iloc[i-1] 
                     else 'red' for i in range(1, len(price_df))]
            colors = ['gray'] + colors
            ax2.bar(price_df.index, price_df['volume'], color=colors, alpha=0.5)
            ax2.set_ylabel('Volume')
        
        # Composite score
        ax3 = axes[2]
        if 'composite_score' in signals_df.columns:
            ax3.fill_between(signals_df.index, signals_df['composite_score'], 
                            alpha=0.5, color=self.colors['signal'])
            ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='High threshold')
            ax3.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Low threshold')
            ax3.set_ylabel('Composite Score')
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_anomaly_analysis(self,
                              price_df: pd.DataFrame,
                              anomaly_df: pd.DataFrame,
                              save_path: str = None) -> plt.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            price_df: Price DataFrame
            anomaly_df: Anomaly detection results
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                  gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # Price with anomaly highlights
        ax1 = axes[0]
        ax1.plot(price_df.index, price_df['close'], color=self.colors['price'], 
                 linewidth=1.5, label='Price')
        
        # Highlight anomaly periods
        if 'stat_anomaly_ratio' in anomaly_df.columns:
            high_anomaly = anomaly_df['stat_anomaly_ratio'] > 0.3
            for i in range(len(high_anomaly)):
                if high_anomaly.iloc[i]:
                    ax1.axvspan(anomaly_df.index[i], 
                               anomaly_df.index[min(i+1, len(anomaly_df)-1)],
                               alpha=0.2, color=self.colors['anomaly'])
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price with Anomaly Highlights', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Z-score
        ax2 = axes[1]
        if 'returns_zscore' in anomaly_df.columns:
            ax2.plot(anomaly_df.index, anomaly_df['returns_zscore'], 
                    color=self.colors['anomaly'], linewidth=1)
            ax2.axhline(y=2.5, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=-2.5, color='red', linestyle='--', alpha=0.7)
            ax2.fill_between(anomaly_df.index, anomaly_df['returns_zscore'], 0, 
                            alpha=0.3, color=self.colors['anomaly'])
            ax2.set_ylabel('Return Z-Score')
            ax2.set_ylim(-4, 4)
        
        # Volume anomaly
        ax3 = axes[2]
        if 'volume_anomaly_score' in anomaly_df.columns:
            ax3.plot(anomaly_df.index, anomaly_df['volume_anomaly_score'],
                    color=self.colors['volume'], linewidth=1)
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
            ax3.set_ylabel('Volume Anomaly')
        
        # ML ensemble score
        ax4 = axes[3]
        if 'ml_ensemble_score' in anomaly_df.columns:
            ax4.plot(anomaly_df.index, anomaly_df['ml_ensemble_score'],
                    color='purple', linewidth=1)
            ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
            ax4.set_ylabel('ML Anomaly Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_backtest_results(self,
                              backtest_result,
                              benchmark_returns: pd.Series = None,
                              save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive backtest results.
        
        Args:
            backtest_result: BacktestResult object
            benchmark_returns: Optional benchmark returns
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.5))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity = backtest_result.equity_curve
        ax1.plot(equity.index, equity, color=self.colors['equity'], 
                 linewidth=2, label='Strategy')
        
        # Benchmark
        if benchmark_returns is not None:
            benchmark_equity = 100000 * (1 + benchmark_returns).cumprod()
            benchmark_equity = benchmark_equity.reindex(equity.index, method='ffill')
            ax1.plot(benchmark_equity.index, benchmark_equity, 
                    color=self.colors['benchmark'], linewidth=1.5, 
                    linestyle='--', label='Buy & Hold')
        
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, 
                        color=self.colors['drawdown'], alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown', fontsize=12)
        
        # Monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_monthly_returns(backtest_result.daily_returns, ax3)
        
        # Trade distribution
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_trade_distribution(backtest_result.trades, ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _plot_monthly_returns(self, daily_returns: pd.Series, ax):
        """Plot monthly returns as text table."""
        monthly = daily_returns.resample('M').sum() * 100
        
        # Create simple text display
        ax.axis('off')
        ax.set_title('Monthly Returns (%)', fontsize=12)
        
        # Show last 12 months
        recent = monthly.tail(12)
        text = '\n'.join([f"{d.strftime('%Y-%m')}: {v:+.1f}%" 
                         for d, v in recent.items()])
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace')
    
    def _plot_trade_distribution(self, trades, ax):
        """Plot trade return distribution."""
        if not trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center', transform=ax.transAxes)
            return
        
        returns = [t.pnl * 100 for t in trades]
        
        ax.hist(returns, bins=20, color=self.colors['equity'], alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}%')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Trade Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade Distribution', fontsize=12)
        ax.legend()
    
    def plot_cyclical_analysis(self,
                               price_df: pd.DataFrame,
                               cyclical_df: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
        """
        Plot cyclical analysis results.
        
        Args:
            price_df: Price DataFrame
            cyclical_df: Cyclical analysis results
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Price with Bollinger Bands
        ax1 = axes[0]
        ax1.plot(price_df.index, price_df['close'], color=self.colors['price'], 
                 linewidth=1.5, label='Price')
        
        if 'bb_middle' in price_df.columns:
            ax1.plot(price_df.index, price_df['bb_middle'], color='orange', 
                    linewidth=1, alpha=0.7, label='BB Middle')
            ax1.fill_between(price_df.index, price_df['bb_lower'], price_df['bb_upper'],
                            alpha=0.2, color='orange')
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price with Bollinger Bands', fontsize=12)
        ax1.legend()
        
        # RSI
        ax2 = axes[1]
        if 'rsi' in cyclical_df.columns:
            ax2.plot(cyclical_df.index, cyclical_df['rsi'], color='purple', linewidth=1)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.fill_between(cyclical_df.index, cyclical_df['rsi'], 50, alpha=0.3)
            ax2.set_ylabel('RSI')
            ax2.set_ylim(0, 100)
        
        # Mean reversion signal
        ax3 = axes[2]
        if 'ou_mr_signal' in cyclical_df.columns:
            ax3.plot(cyclical_df.index, cyclical_df['ou_mr_signal'], 
                    color='teal', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.fill_between(cyclical_df.index, cyclical_df['ou_mr_signal'], 0, 
                            alpha=0.3, color='teal')
            ax3.set_ylabel('Mean Reversion Signal')
        
        # Spectral entropy (market complexity)
        ax4 = axes[3]
        if 'spectral_entropy' in cyclical_df.columns:
            ax4.plot(cyclical_df.index, cyclical_df['spectral_entropy'], 
                    color='brown', linewidth=1)
            ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High complexity')
            ax4.set_ylabel('Spectral Entropy')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_summary_report(self,
                                backtest_result,
                                analysis_summary: Dict,
                                save_path: str = None) -> str:
        """
        Generate text summary report.
        
        Args:
            backtest_result: BacktestResult object
            analysis_summary: Analysis summary dictionary
            save_path: Path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("STOCK ANOMALY PREDICTION SYSTEM - BACKTEST REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Return:        {backtest_result.total_return:>10.2%}")
        report.append(f"Annual Return:       {backtest_result.annual_return:>10.2%}")
        report.append(f"Sharpe Ratio:        {backtest_result.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:       {backtest_result.sortino_ratio:>10.2f}")
        report.append(f"Max Drawdown:        {backtest_result.max_drawdown:>10.2%}")
        report.append(f"Calmar Ratio:        {backtest_result.calmar_ratio:>10.2f}")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:        {backtest_result.total_trades:>10}")
        report.append(f"Win Rate:            {backtest_result.win_rate:>10.1%}")
        report.append(f"Profit Factor:       {backtest_result.profit_factor:>10.2f}")
        report.append(f"Avg Trade Return:    {backtest_result.avg_trade_return:>10.2%}")
        report.append(f"Avg Holding Period:  {backtest_result.avg_holding_period:>10.1f} days")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Annual Volatility:   {backtest_result.volatility:>10.2%}")
        report.append(f"Skewness:            {backtest_result.skewness:>10.2f}")
        report.append(f"Kurtosis:            {backtest_result.kurtosis:>10.2f}")
        report.append(f"VaR (95%):           {backtest_result.var_95:>10.2%}")
        report.append(f"CVaR (95%):          {backtest_result.cvar_95:>10.2%}")
        report.append("")
        
        # Current Analysis
        if analysis_summary:
            report.append("CURRENT ANALYSIS")
            report.append("-" * 40)
            for key, value in analysis_summary.items():
                if isinstance(value, float):
                    report.append(f"{key:<25}{value:>10.3f}")
                else:
                    report.append(f"{key:<25}{str(value):>10}")
        
        report.append("")
        report.append("=" * 70)
        report.append("Generated by Stock Anomaly Prediction System")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved: {save_path}")
        
        return report_text
    
    def save_all_charts(self,
                        price_df: pd.DataFrame,
                        signals_df: pd.DataFrame,
                        anomaly_df: pd.DataFrame,
                        cyclical_df: pd.DataFrame,
                        backtest_result,
                        output_dir: str = 'output'):
        """
        Generate and save all charts.
        
        Args:
            price_df: Price DataFrame
            signals_df: Signals DataFrame
            anomaly_df: Anomaly DataFrame
            cyclical_df: Cyclical DataFrame
            backtest_result: BacktestResult
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating charts in {output_dir}/...")
        
        # Price with signals
        self.plot_price_with_signals(
            price_df, signals_df,
            save_path=f"{output_dir}/price_signals.png"
        )
        
        # Anomaly analysis
        self.plot_anomaly_analysis(
            price_df, anomaly_df,
            save_path=f"{output_dir}/anomaly_analysis.png"
        )
        
        # Backtest results
        benchmark = price_df['returns'] if 'returns' in price_df.columns else None
        self.plot_backtest_results(
            backtest_result, benchmark,
            save_path=f"{output_dir}/backtest_results.png"
        )
        
        # Cyclical analysis
        self.plot_cyclical_analysis(
            price_df, cyclical_df,
            save_path=f"{output_dir}/cyclical_analysis.png"
        )
        
        print("All charts generated!")
        plt.close('all')


if __name__ == "__main__":
    print("Testing Dashboard Generator...")
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    price_df = pd.DataFrame({
        'close': 10 + np.cumsum(np.random.randn(252) * 0.1),
        'volume': np.random.randint(1000000, 5000000, 252),
        'returns': np.random.randn(252) * 0.02
    }, index=dates)
    
    signals_df = pd.DataFrame({
        'buy_signal': np.random.random(252) < 0.05,
        'composite_score': 50 + np.cumsum(np.random.randn(252) * 2)
    }, index=dates)
    signals_df['composite_score'] = signals_df['composite_score'].clip(0, 100)
    
    # Generate dashboard
    dashboard = DashboardGenerator()
    fig = dashboard.plot_price_with_signals(price_df, signals_df)
    plt.show()
