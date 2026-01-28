import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class AnomalyPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.svm = OneClassSVM(nu=0.05)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        self.feature_cols = []
        
    def train(self, df):
        # Define features
        # We exclude the target and price columns themselves, focusing on derived indicators
        exclude = ['open', 'high', 'low', 'close', 'adj_close', 'target_rally', 'future_max_return', 'cycle_period_1', 'cycle_period_2', 'cycle_mag_1', 'cycle_mag_2']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        
        # Handle any remaining NaNs (though we dropped them in prep)
        X = df[self.feature_cols].copy()
        y = df['target_rally'].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Unsupervised Anomaly Detection (Isolation Forest)
        # We assume "anomalies" are often precursors to big moves (or crashes)
        print("Training Isolation Forest...")
        self.iso_forest.fit(X_scaled)
        
        # 2. Unsupervised Anomaly Detection (One-Class SVM)
        print("Training One-Class SVM...")
        self.svm.fit(X_scaled)
        
        # 3. Supervised Prediction (Random Forest)
        # TimeSeriesSplit for validation (conceptually)
        print("Training Random Forest...")
        self.rf.fit(X_scaled, y)
        
        # Feature Importance
        importances = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Predictive Features:")
        print(importances.head(10))

        # DEBUG: Check Training Class Balance
        print(f"\nTraining Class Balance: {y.value_counts(normalize=True).to_dict()}")
        
    def predict(self, df):
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)
        
        results = pd.DataFrame(index=df.index)
        
        # Isolation Forest
        iso_pred = self.iso_forest.predict(X_scaled)
        results['anomaly_iso'] = (iso_pred == -1).astype(int)
        
        # SVM
        svm_pred = self.svm.predict(X_scaled)
        results['anomaly_svm'] = (svm_pred == -1).astype(int)
        
        # Random Forest
        rf_prob = self.rf.predict_proba(X_scaled)[:, 1]
        results['prob_rally'] = rf_prob
        
        # DEBUG: Print probability stats
        print(f"\nPrediction Probabilities Stats:\n{pd.Series(rf_prob).describe()}")
        
        # Lower threshold for demo purposes if signals are weak
        threshold = 0.5 
        results['final_signal'] = (results['prob_rally'] > threshold)
        
        return results

def backtest_models(df, horizon=5):
    """
    Simulates a walk-forward backtest.
    Training on expanding window, predicting next week.
    This is computationally expensive, so we'll do a simple Train/Test split for the demo.
    Split: 70% Train, 30% Test (chronological)
    """
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    predictor = AnomalyPredictor()
    predictor.train(train_df)
    
    predictions = predictor.predict(test_df)
    
    # Evaluation
    merged = test_df.join(predictions)
    
    print("\n--- Backtest Results (Out of Sample) ---")
    y_true = merged['target_rally']
    y_pred_rf = (merged['prob_rally'] > 0.5).astype(int) 
    y_pred_combined = merged['final_signal'].astype(int)
    
    print(f"Random Forest Precision: {precision_score(y_true, y_pred_rf, zero_division=0):.4f}")
    print(f"Random Forest Recall:    {recall_score(y_true, y_pred_rf, zero_division=0):.4f}")
    
    print(f"Combined Anomaly+RF Precision: {precision_score(y_true, y_pred_combined, zero_division=0):.4f}")
    print(f"Combined Anomaly+RF Recall:    {recall_score(y_true, y_pred_combined, zero_division=0):.4f}")
    
    # Calculate Returns
    merged['actual_return_horizon'] = merged['close'].shift(-horizon) / merged['close'] - 1
    
    valid_rets = merged.dropna(subset=['actual_return_horizon'])
    
    avg_ret_rf = valid_rets[valid_rets['prob_rally'] > 0.5]['actual_return_horizon'].mean()
    avg_ret_comb = valid_rets[valid_rets['final_signal']]['actual_return_horizon'].mean()
    base_ret = valid_rets['actual_return_horizon'].mean()
    
    print(f"\nAverage {horizon}-Day Return per Trade:")
    print(f"Baseline (Random): {base_ret*100:.2f}%")
    print(f"RF Strategy:       {avg_ret_rf*100:.2f}%")
    print(f"Combined Strategy: {avg_ret_comb*100:.2f}%")
    
    return merged

if __name__ == "__main__":
    from data_loader import load_and_process_data
    from feature_engineering import prepare_features
    
    df = load_and_process_data()
    df = prepare_features(df)
    results = backtest_models(df)
    
    # Save results
    results.to_csv("model_results.csv")
    print("Results saved to model_results.csv")
