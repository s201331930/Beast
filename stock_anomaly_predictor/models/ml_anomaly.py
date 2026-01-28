"""
Machine Learning Anomaly Detection Module

Advanced ML-based anomaly detection methods:
- Isolation Forest (tree-based isolation)
- One-Class SVM (kernel-based boundary)
- Local Outlier Factor (density-based)
- DBSCAN (clustering-based)
- Autoencoder (reconstruction error)
- LSTM Autoencoder (temporal patterns)
- Variational Autoencoder (probabilistic)
- Gaussian Mixture Models (probabilistic clustering)
- XGBoost Anomaly Scoring

Mathematical Foundation: Combines unsupervised learning with probabilistic modeling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class MLAnomalyDetector:
    """
    Machine Learning based anomaly detection for financial data.
    """
    
    def __init__(self, df: pd.DataFrame, feature_columns: List[str] = None):
        """
        Initialize ML anomaly detector.
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use as features
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.anomalies = pd.DataFrame(index=df.index)
        
        # Default feature columns if not specified
        if self.feature_columns is None:
            self.feature_columns = self._get_default_features()
        
        self._prepare_features()
    
    def _get_default_features(self) -> List[str]:
        """Get default feature columns for anomaly detection."""
        potential_features = [
            'returns', 'log_returns', 'volume_change',
            'high_low_range', 'close_open_range', 'gap',
            'rsi', 'macd', 'macd_histogram',
            'bb_position', 'bb_width',
            'stoch_k', 'stoch_d',
            'volume_ratio', 'atr_percent',
            'roc_5', 'roc_10', 'roc_20',
            'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_ratio',
            'obv_divergence', 'rsi_divergence'
        ]
        
        return [f for f in potential_features if f in self.df.columns]
    
    def _prepare_features(self):
        """Prepare and scale features for ML models."""
        # Get available features
        available_features = [f for f in self.feature_columns if f in self.df.columns]
        
        if len(available_features) < 3:
            print(f"Warning: Only {len(available_features)} features available")
            # Add basic features if needed
            if 'returns' in self.df.columns:
                available_features.append('returns')
        
        self.feature_columns = available_features
        
        # Create feature matrix
        self.X = self.df[self.feature_columns].copy()
        
        # Handle missing values
        self.X = self.X.ffill().bfill().fillna(0)
        
        # Replace infinities
        self.X = self.X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            index=self.X.index,
            columns=self.X.columns
        )
        
        print(f"Prepared {len(self.feature_columns)} features for ML models")
    
    def isolation_forest(self,
                         contamination: float = None,
                         n_estimators: int = 200,
                         random_state: int = 42) -> pd.Series:
        """
        Isolation Forest anomaly detection.
        
        Mathematical basis:
        - Randomly selects a feature and split value
        - Anomalies are isolated quickly (short path length)
        - Anomaly score = 2^(-E(h(x))/c(n))
        
        Advantages: Works well with high-dimensional data, fast
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
            random_state: Random seed
            
        Returns:
            Series of anomaly scores (-1 to 1, lower = more anomalous)
        """
        contamination = contamination or config.anomaly.isolation_forest_contamination
        
        print("Running Isolation Forest...")
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Fit and predict
        predictions = model.fit_predict(self.X_scaled)
        scores = model.decision_function(self.X_scaled)
        
        self.anomalies['isolation_forest_score'] = scores
        self.anomalies['isolation_forest_anomaly'] = predictions == -1
        
        n_anomalies = (predictions == -1).sum()
        print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(predictions):.2f}%)")
        
        return pd.Series(scores, index=self.X_scaled.index)
    
    def one_class_svm(self,
                      nu: float = 0.05,
                      kernel: str = 'rbf',
                      gamma: str = 'scale') -> pd.Series:
        """
        One-Class SVM for anomaly detection.
        
        Mathematical basis:
        - Finds a hyperplane that separates data from origin
        - Uses kernel trick for non-linear boundaries
        - Minimizes: 1/2 ||w||² + 1/(νn) Σξ_i - ρ
        
        Advantages: Can capture complex boundaries
        
        Args:
            nu: Upper bound on fraction of outliers
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: Kernel coefficient
            
        Returns:
            Series of decision function values
        """
        print("Running One-Class SVM...")
        
        # Use subset for speed if data is large
        if len(self.X_scaled) > 5000:
            # Train on sample, predict on all
            sample_idx = np.random.choice(len(self.X_scaled), 5000, replace=False)
            X_train = self.X_scaled.iloc[sample_idx]
        else:
            X_train = self.X_scaled
        
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        model.fit(X_train)
        
        predictions = model.predict(self.X_scaled)
        scores = model.decision_function(self.X_scaled)
        
        self.anomalies['ocsvm_score'] = scores
        self.anomalies['ocsvm_anomaly'] = predictions == -1
        
        n_anomalies = (predictions == -1).sum()
        print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(predictions):.2f}%)")
        
        return pd.Series(scores, index=self.X_scaled.index)
    
    def local_outlier_factor(self,
                             n_neighbors: int = None,
                             contamination: float = None) -> pd.Series:
        """
        Local Outlier Factor (LOF) for density-based anomaly detection.
        
        Mathematical basis:
        LOF(x) = Σ_{y∈N_k(x)} [lrd(y) / lrd(x)] / |N_k(x)|
        
        Where lrd (local reachability density) measures local density.
        Points with LOF >> 1 are anomalies (lower density than neighbors)
        
        Advantages: Captures local structure, works with clusters
        
        Args:
            n_neighbors: Number of neighbors for density estimation
            contamination: Expected outlier fraction
            
        Returns:
            Series of LOF scores (negative = more anomalous)
        """
        n_neighbors = n_neighbors or config.anomaly.lof_neighbors
        contamination = contamination or config.anomaly.isolation_forest_contamination
        
        print("Running Local Outlier Factor...")
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(self.X_scaled)
        scores = model.negative_outlier_factor_
        
        self.anomalies['lof_score'] = scores
        self.anomalies['lof_anomaly'] = predictions == -1
        
        n_anomalies = (predictions == -1).sum()
        print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(predictions):.2f}%)")
        
        return pd.Series(scores, index=self.X_scaled.index)
    
    def dbscan_anomaly(self,
                       eps: float = None,
                       min_samples: int = 5) -> pd.Series:
        """
        DBSCAN clustering for anomaly detection.
        
        Mathematical basis:
        - Core points: Have >= min_samples within eps distance
        - Border points: Within eps of core point
        - Noise points: Neither (these are anomalies)
        
        Advantages: No assumptions about cluster shape
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples for core point
            
        Returns:
            Series of cluster labels (-1 = noise/anomaly)
        """
        eps = eps or config.anomaly.dbscan_eps
        
        print("Running DBSCAN clustering...")
        
        # Reduce dimensionality for DBSCAN
        pca = PCA(n_components=min(10, len(self.feature_columns)))
        X_pca = pca.fit_transform(self.X_scaled)
        
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_pca)
        
        self.anomalies['dbscan_label'] = labels
        self.anomalies['dbscan_anomaly'] = labels == -1
        
        n_anomalies = (labels == -1).sum()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Found {n_clusters} clusters, {n_anomalies} noise points ({100*n_anomalies/len(labels):.2f}%)")
        
        return pd.Series(labels, index=self.X_scaled.index)
    
    def gaussian_mixture_anomaly(self,
                                 n_components: int = 5,
                                 threshold_percentile: float = 5) -> pd.Series:
        """
        Gaussian Mixture Model for probabilistic anomaly detection.
        
        Mathematical basis:
        p(x) = Σ_k π_k N(x|μ_k, Σ_k)
        
        Anomaly score = -log p(x) (log-likelihood)
        Low probability = anomaly
        
        Advantages: Probabilistic interpretation, captures multimodal distributions
        
        Args:
            n_components: Number of Gaussian components
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Series of negative log-likelihood scores
        """
        print("Running Gaussian Mixture Model...")
        
        model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42
        )
        
        model.fit(self.X_scaled)
        
        # Score samples (log-likelihood)
        log_likelihood = model.score_samples(self.X_scaled)
        
        # Lower log-likelihood = more anomalous
        threshold = np.percentile(log_likelihood, threshold_percentile)
        
        self.anomalies['gmm_score'] = log_likelihood
        self.anomalies['gmm_anomaly'] = log_likelihood < threshold
        
        n_anomalies = (log_likelihood < threshold).sum()
        print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(log_likelihood):.2f}%)")
        
        return pd.Series(log_likelihood, index=self.X_scaled.index)
    
    def rolling_anomaly_scores(self,
                               window: int = 60,
                               method: str = 'isolation_forest') -> pd.Series:
        """
        Calculate rolling anomaly scores using a sliding window approach.
        
        This captures concept drift and regime changes.
        
        Args:
            window: Rolling window size
            method: Anomaly detection method to use
            
        Returns:
            Series of rolling anomaly scores
        """
        print(f"Running rolling {method} with window={window}...")
        
        scores = pd.Series(index=self.X_scaled.index, dtype=float)
        
        for i in range(window, len(self.X_scaled)):
            X_window = self.X_scaled.iloc[i-window:i]
            X_current = self.X_scaled.iloc[i:i+1]
            
            try:
                if method == 'isolation_forest':
                    model = IsolationForest(contamination=0.1, random_state=42)
                    model.fit(X_window)
                    scores.iloc[i] = model.decision_function(X_current)[0]
                    
                elif method == 'lof':
                    model = LocalOutlierFactor(n_neighbors=min(20, window//3), novelty=True)
                    model.fit(X_window)
                    scores.iloc[i] = model.decision_function(X_current)[0]
                    
            except Exception as e:
                scores.iloc[i] = np.nan
        
        self.anomalies[f'rolling_{method}_score'] = scores
        
        return scores
    
    def autoencoder_anomaly(self,
                            encoding_dim: int = 8,
                            epochs: int = 50,
                            threshold_percentile: float = None) -> pd.Series:
        """
        Autoencoder-based anomaly detection using reconstruction error.
        
        Mathematical basis:
        - Encoder: h = f(Wx + b)
        - Decoder: x' = g(W'h + b')
        - Loss: ||x - x'||² (reconstruction error)
        
        High reconstruction error = anomaly
        
        Advantages: Learns compressed representation, captures complex patterns
        
        Args:
            encoding_dim: Dimension of encoded representation
            epochs: Training epochs
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Series of reconstruction errors
        """
        threshold_percentile = threshold_percentile or config.anomaly.autoencoder_threshold_percentile
        
        print("Running Autoencoder anomaly detection...")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            tf.get_logger().setLevel('ERROR')
            
            input_dim = len(self.feature_columns)
            
            # Build autoencoder
            encoder_input = keras.Input(shape=(input_dim,))
            x = layers.Dense(64, activation='relu')(encoder_input)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            encoded = layers.Dense(encoding_dim, activation='relu')(x)
            
            x = layers.Dense(32, activation='relu')(encoded)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            decoded = layers.Dense(input_dim, activation='linear')(x)
            
            autoencoder = keras.Model(encoder_input, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train
            X_train = self.X_scaled.values
            autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Get reconstruction error
            reconstructed = autoencoder.predict(X_train, verbose=0)
            mse = np.mean(np.power(X_train - reconstructed, 2), axis=1)
            
            threshold = np.percentile(mse, threshold_percentile)
            
            self.anomalies['autoencoder_mse'] = mse
            self.anomalies['autoencoder_anomaly'] = mse > threshold
            
            n_anomalies = (mse > threshold).sum()
            print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(mse):.2f}%)")
            
            return pd.Series(mse, index=self.X_scaled.index)
            
        except ImportError:
            print("  TensorFlow not available, skipping autoencoder")
            return pd.Series(index=self.X_scaled.index)
    
    def lstm_autoencoder_anomaly(self,
                                 sequence_length: int = 10,
                                 encoding_dim: int = 16,
                                 epochs: int = 30,
                                 threshold_percentile: float = None) -> pd.Series:
        """
        LSTM Autoencoder for temporal anomaly detection.
        
        Captures temporal dependencies in the sequence of observations.
        
        Args:
            sequence_length: Length of input sequences
            encoding_dim: LSTM hidden dimension
            epochs: Training epochs
            threshold_percentile: Percentile for threshold
            
        Returns:
            Series of reconstruction errors
        """
        threshold_percentile = threshold_percentile or config.anomaly.autoencoder_threshold_percentile
        
        print("Running LSTM Autoencoder anomaly detection...")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            tf.get_logger().setLevel('ERROR')
            
            # Prepare sequences
            X = self.X_scaled.values
            sequences = []
            indices = []
            
            for i in range(len(X) - sequence_length):
                sequences.append(X[i:i+sequence_length])
                indices.append(self.X_scaled.index[i+sequence_length-1])
            
            sequences = np.array(sequences)
            n_features = sequences.shape[2]
            
            # Build LSTM autoencoder
            encoder_input = keras.Input(shape=(sequence_length, n_features))
            x = layers.LSTM(32, activation='relu', return_sequences=True)(encoder_input)
            x = layers.LSTM(encoding_dim, activation='relu', return_sequences=False)(x)
            
            x = layers.RepeatVector(sequence_length)(x)
            x = layers.LSTM(encoding_dim, activation='relu', return_sequences=True)(x)
            x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
            decoded = layers.TimeDistributed(layers.Dense(n_features))(x)
            
            model = keras.Model(encoder_input, decoded)
            model.compile(optimizer='adam', loss='mse')
            
            # Train
            model.fit(
                sequences, sequences,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Get reconstruction error
            reconstructed = model.predict(sequences, verbose=0)
            mse = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
            
            threshold = np.percentile(mse, threshold_percentile)
            
            # Create series with proper index
            mse_series = pd.Series(index=self.X_scaled.index, dtype=float)
            for i, idx in enumerate(indices):
                mse_series[idx] = mse[i]
            
            self.anomalies['lstm_ae_mse'] = mse_series
            self.anomalies['lstm_ae_anomaly'] = mse_series > threshold
            
            n_anomalies = (mse_series > threshold).sum()
            print(f"  Detected {n_anomalies} anomalies")
            
            return mse_series
            
        except ImportError:
            print("  TensorFlow not available, skipping LSTM autoencoder")
            return pd.Series(index=self.X_scaled.index)
    
    def ensemble_score(self, weights: Dict[str, float] = None) -> pd.Series:
        """
        Create ensemble anomaly score from all methods.
        
        Args:
            weights: Dictionary of method weights
            
        Returns:
            Combined ensemble anomaly score
        """
        if weights is None:
            weights = {
                'isolation_forest_score': 0.25,
                'lof_score': 0.20,
                'ocsvm_score': 0.15,
                'gmm_score': 0.15,
                'autoencoder_mse': 0.15,
                'lstm_ae_mse': 0.10
            }
        
        # Normalize scores to [0, 1] range
        normalized_scores = pd.DataFrame(index=self.anomalies.index)
        
        for col in weights.keys():
            if col in self.anomalies.columns:
                scores = self.anomalies[col].dropna()
                if len(scores) > 0:
                    # Min-max normalization
                    min_val = scores.min()
                    max_val = scores.max()
                    if max_val > min_val:
                        # For scores where higher = more anomalous (MSE)
                        if 'mse' in col:
                            normalized_scores[col] = (self.anomalies[col] - min_val) / (max_val - min_val)
                        # For scores where lower = more anomalous
                        else:
                            normalized_scores[col] = 1 - (self.anomalies[col] - min_val) / (max_val - min_val)
        
        # Weighted average
        ensemble_score = pd.Series(0.0, index=self.anomalies.index)
        total_weight = 0
        
        for col, weight in weights.items():
            if col in normalized_scores.columns:
                ensemble_score += normalized_scores[col].fillna(0) * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        self.anomalies['ml_ensemble_score'] = ensemble_score
        self.anomalies['ml_ensemble_anomaly'] = ensemble_score > 0.7
        
        return ensemble_score
    
    def run_all_detectors(self, include_deep_learning: bool = True) -> pd.DataFrame:
        """
        Run all ML anomaly detectors.
        
        Args:
            include_deep_learning: Whether to include autoencoder methods
            
        Returns:
            DataFrame with all anomaly scores
        """
        print("=" * 60)
        print("Running ML Anomaly Detection Suite")
        print("=" * 60)
        
        # Traditional ML methods
        self.isolation_forest()
        self.one_class_svm()
        self.local_outlier_factor()
        self.dbscan_anomaly()
        self.gaussian_mixture_anomaly()
        
        # Deep learning methods
        if include_deep_learning:
            self.autoencoder_anomaly()
            self.lstm_autoencoder_anomaly()
        
        # Ensemble
        self.ensemble_score()
        
        print("=" * 60)
        print("ML Anomaly Detection Complete")
        
        return self.anomalies
    
    def get_top_anomalies(self, n: int = 20, method: str = 'ml_ensemble_score') -> pd.DataFrame:
        """
        Get top N most anomalous observations.
        
        Args:
            n: Number of top anomalies
            method: Score column to use
            
        Returns:
            DataFrame of top anomalies
        """
        if method not in self.anomalies.columns:
            method = 'isolation_forest_score'
        
        # Sort by score (ascending for most methods since lower = more anomalous)
        if 'mse' in method or method == 'ml_ensemble_score':
            sorted_df = self.anomalies.sort_values(method, ascending=False)
        else:
            sorted_df = self.anomalies.sort_values(method, ascending=True)
        
        return sorted_df.head(n)


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, '..')
    from data.collector import DataCollector
    
    # Collect data
    collector = DataCollector("RKLB")
    data = collector.collect_all_data()
    df = data['primary']
    
    # Run ML anomaly detection
    detector = MLAnomalyDetector(df)
    anomalies = detector.run_all_detectors(include_deep_learning=True)
    
    # Get top anomalies
    print("\nTop 10 Anomalies:")
    print(detector.get_top_anomalies(10))
