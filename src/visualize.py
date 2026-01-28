import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results_file="model_results.csv"):
    df = pd.read_csv(results_file, parse_dates=True, index_col=0)
    
    plt.figure(figsize=(14, 8))
    
    # Plot Price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.6)
    
    # Plot Signals
    # Buy Signals (Final Signal)
    buy_signals = df[df['final_signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Model Buy Signal')
    
    ax1.set_title('RKLB Price & Model Signals')
    ax1.legend()
    
    # Plot Probability
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['prob_rally'], label='Rally Probability', color='blue')
    ax2.axhline(0.5, color='red', linestyle='--', label='Threshold')
    ax2.set_title('Model Confidence (Probability of Rally)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('backtest_plot.png')
    print("Plot saved to backtest_plot.png")

if __name__ == "__main__":
    plot_results()
