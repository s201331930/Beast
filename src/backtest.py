import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, df, signal_col, future_window=5, move_threshold=0.05):
        self.df = df
        self.signal_col = signal_col
        self.future_window = future_window
        self.move_threshold = move_threshold

    def label_actual_big_moves(self):
        """
        Labels days followed by a big move in the next 'future_window' days.
        """
        # Max price in next N days
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.future_window)
        future_max = self.df['Close'].rolling(window=indexer).max()
        
        # Calculate percentage change from current close to future max
        max_return = (future_max - self.df['Close']) / self.df['Close']
        
        self.df['Is_Big_Move'] = (max_return > self.move_threshold).astype(int)
        return self.df['Is_Big_Move']

    def evaluate(self):
        """
        Evaluates the signals against actual big moves.
        """
        self.label_actual_big_moves()
        
        # Confusion Matrix
        tp = ((self.df[self.signal_col] == 1) & (self.df['Is_Big_Move'] == 1)).sum()
        fp = ((self.df[self.signal_col] == 1) & (self.df['Is_Big_Move'] == 0)).sum()
        fn = ((self.df[self.signal_col] == 0) & (self.df['Is_Big_Move'] == 1)).sum()
        tn = ((self.df[self.signal_col] == 0) & (self.df['Is_Big_Move'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"--- Backtest Results for {self.signal_col} ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def simulate_trading(self, initial_capital=10000):
        """
        Simple trading strategy: Buy on signal, hold for future_window days.
        """
        capital = initial_capital
        position = 0
        equity_curve = [capital]
        
        dates = self.df.index
        signals = self.df[self.signal_col].values
        prices = self.df['Close'].values
        
        # Simple non-overlapping trades for demonstration
        i = 0
        while i < len(self.df) - self.future_window:
            if signals[i] == 1:
                # Buy
                entry_price = prices[i]
                shares = capital / entry_price
                
                # Sell after window
                exit_price = prices[i + self.future_window]
                capital = shares * exit_price
                
                # print(f"Trade at {dates[i]}: Buy {entry_price:.2f}, Sell {exit_price:.2f}, New Capital: {capital:.2f}")
                
                # Skip the holding period
                for _ in range(self.future_window):
                    equity_curve.append(capital)
                i += self.future_window
            else:
                equity_curve.append(capital)
                i += 1
                
        # Fill remaining
        while len(equity_curve) < len(self.df):
            equity_curve.append(capital)
            
        self.df[f'{self.signal_col}_Equity'] = equity_curve
        return capital

    def plot_results(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Close'], label='Price', alpha=0.5)
        
        # Plot signals
        signals = self.df[self.df[self.signal_col] == 1]
        plt.scatter(signals.index, signals['Close'], color='red', label='Anomaly Signal', marker='^')
        
        plt.title(f"Anomaly Detection Signals: {self.signal_col}")
        plt.legend()
        plt.savefig(f"data/{self.signal_col}_chart.png")
        print(f"Chart saved to data/{self.signal_col}_chart.png")

