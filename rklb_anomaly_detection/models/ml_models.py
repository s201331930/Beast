"""
Machine Learning Anomaly Detection Models
==========================================
Implements ML-based methods for detecting anomalies in financial time series:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoder (Neural Network)
- DBSCAN clustering
- Gaussian Mixture Models
- Matrix Profile (STUMPY)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import Dict, Optional, List, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class IsolationForestDetector:
    """
    Isolation Forest anomaly detection.
    Isolates anomalies by randomly selecting features and split values.
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, 
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.name = "IsoForest"
        self.model = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Isolation Forest and detect anomalies.
        """
        # Prepare features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < 100:
            logger.warning("Insufficient data for Isolation Forest")
            return pd.DataFrame(index=data.index)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df)
        
        # Fit model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Predictions: -1 for anomalies, 1 for normal
        predictions = self.model.fit_predict(X_scaled)
        
        # Anomaly scores (more negative = more anomalous)
        scores = self.model.decision_function(X_scaled)
        
        # Normalize scores to [0, 1] where higher = more anomalous
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_prediction': predictions,
            f'{self.name}_anomaly': (predictions == -1).astype(int),
            f'{self.name}_raw_score': scores,
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)
    
    def rolling_detection(self, data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Rolling window Isolation Forest detection for adaptive anomaly detection.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        results = []
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            current = df.iloc[i:i+1]
            
            # Scale using window statistics
            scaler = StandardScaler()
            window_scaled = scaler.fit_transform(window_data)
            current_scaled = scaler.transform(current)
            
            # Train on window, predict on current
            model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
            model.fit(window_scaled)
            
            score = model.decision_function(current_scaled)[0]
            pred = model.predict(current_scaled)[0]
            
            results.append({
                'score': score,
                'anomaly': pred == -1
            })
        
        result_df = pd.DataFrame(results, index=df.index[window:])
        result_df.columns = [f'{self.name}_rolling_{c}' for c in result_df.columns]
        
        return result_df.reindex(data.index)


class LOFDetector:
    """
    Local Outlier Factor (LOF) detection.
    Measures local density deviation of a point with respect to neighbors.
    """
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.name = "LOF"
        self.scaler = StandardScaler()
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit LOF model and detect anomalies.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < self.n_neighbors + 1:
            logger.warning("Insufficient data for LOF")
            return pd.DataFrame(index=data.index)
        
        X_scaled = self.scaler.fit_transform(df)
        
        # LOF with novelty=False for inductive learning
        model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_
        
        # Normalize scores
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_prediction': predictions,
            f'{self.name}_anomaly': (predictions == -1).astype(int),
            f'{self.name}_raw_score': scores,
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)


class OneClassSVMDetector:
    """
    One-Class SVM for novelty detection.
    Learns a decision boundary around normal data.
    """
    
    def __init__(self, kernel: str = 'rbf', nu: float = 0.05, gamma: str = 'scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.name = "OCSVM"
        self.scaler = StandardScaler()
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit One-Class SVM and detect anomalies.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < 100:
            logger.warning("Insufficient data for One-Class SVM")
            return pd.DataFrame(index=data.index)
        
        X_scaled = self.scaler.fit_transform(df)
        
        model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )
        
        predictions = model.fit_predict(X_scaled)
        scores = model.decision_function(X_scaled)
        
        # Normalize scores
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_prediction': predictions,
            f'{self.name}_anomaly': (predictions == -1).astype(int),
            f'{self.name}_raw_score': scores,
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detection.
    Uses reconstruction error to identify anomalies.
    """
    
    def __init__(self, encoding_dim: int = 8, epochs: int = 100, 
                 threshold_percentile: float = 95):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile
        self.name = "Autoencoder"
        self.scaler = MinMaxScaler()
        self.model = None
        
    def _build_model(self, input_dim: int):
        """Build autoencoder architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
            from tensorflow.keras.regularizers import l2
            
            # Encoder
            inputs = Input(shape=(input_dim,))
            x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            encoded = Dense(self.encoding_dim, activation='relu')(x)
            
            # Decoder
            x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(encoded)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            decoded = Dense(input_dim, activation='sigmoid')(x)
            
            model = Model(inputs, decoded)
            model.compile(optimizer='adam', loss='mse')
            
            return model
            
        except ImportError:
            logger.warning("TensorFlow not available, using simple autoencoder")
            return None
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit autoencoder and detect anomalies based on reconstruction error.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < 200:
            logger.warning("Insufficient data for Autoencoder")
            return self._fallback_pca(data)
        
        X_scaled = self.scaler.fit_transform(df)
        input_dim = X_scaled.shape[1]
        
        self.model = self._build_model(input_dim)
        
        if self.model is None:
            return self._fallback_pca(data)
        
        # Suppress TensorFlow logging
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Train autoencoder
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction error
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        
        # Threshold based on percentile
        threshold = np.percentile(mse, self.threshold_percentile)
        anomalies = mse > threshold
        
        # Normalize scores
        normalized_scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_reconstruction_error': mse,
            f'{self.name}_anomaly': anomalies.astype(int),
            f'{self.name}_threshold': threshold,
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)
    
    def _fallback_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback using PCA reconstruction error."""
        logger.info("Using PCA reconstruction error as fallback")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        X_scaled = self.scaler.fit_transform(df)
        
        # PCA with reduced dimensions
        n_components = min(self.encoding_dim, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        
        # Transform and inverse transform
        transformed = pca.fit_transform(X_scaled)
        reconstructed = pca.inverse_transform(transformed)
        
        # Reconstruction error
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        
        threshold = np.percentile(mse, self.threshold_percentile)
        anomalies = mse > threshold
        normalized_scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_reconstruction_error': mse,
            f'{self.name}_anomaly': anomalies.astype(int),
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)


class DBSCANDetector:
    """
    DBSCAN clustering for density-based anomaly detection.
    Points not belonging to any cluster are anomalies.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.name = "DBSCAN"
        self.scaler = StandardScaler()
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply DBSCAN and identify outliers (cluster = -1).
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < self.min_samples:
            logger.warning("Insufficient data for DBSCAN")
            return pd.DataFrame(index=data.index)
        
        X_scaled = self.scaler.fit_transform(df)
        
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = model.fit_predict(X_scaled)
        
        # Anomalies are points with label -1 (noise)
        anomalies = labels == -1
        
        # Calculate distance to nearest cluster center for scoring
        # (simplified: use distance to mean)
        mean_point = X_scaled.mean(axis=0)
        distances = np.sqrt(np.sum((X_scaled - mean_point) ** 2, axis=1))
        normalized_scores = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_cluster': labels,
            f'{self.name}_anomaly': anomalies.astype(int),
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)


class GMMDetector:
    """
    Gaussian Mixture Model for probabilistic anomaly detection.
    Uses log probability to identify low-probability points.
    """
    
    def __init__(self, n_components: int = 3, threshold_percentile: float = 5):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.name = "GMM"
        self.scaler = StandardScaler()
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit GMM and detect low-probability anomalies.
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data[numeric_cols].dropna()
        
        if len(df) < 100:
            logger.warning("Insufficient data for GMM")
            return pd.DataFrame(index=data.index)
        
        X_scaled = self.scaler.fit_transform(df)
        
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        
        model.fit(X_scaled)
        
        # Log probability
        log_probs = model.score_samples(X_scaled)
        
        # Low probability points are anomalies
        threshold = np.percentile(log_probs, self.threshold_percentile)
        anomalies = log_probs < threshold
        
        # Normalize scores (invert so higher = more anomalous)
        normalized_scores = 1 - (log_probs - log_probs.min()) / (log_probs.max() - log_probs.min() + 1e-10)
        
        result = pd.DataFrame({
            f'{self.name}_log_prob': log_probs,
            f'{self.name}_anomaly': anomalies.astype(int),
            f'{self.name}_score': normalized_scores
        }, index=df.index)
        
        return result.reindex(data.index)


class MatrixProfileDetector:
    """
    Matrix Profile for time series discord detection.
    Uses STUMPY library for efficient computation.
    """
    
    def __init__(self, window_size: int = 20, threshold_percentile: float = 95):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.name = "MatrixProfile"
        
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate matrix profile and detect discords.
        """
        try:
            import stumpy
            
            # Clean series
            clean_series = series.dropna().values.astype(float)
            
            if len(clean_series) < self.window_size * 2:
                logger.warning("Insufficient data for Matrix Profile")
                return pd.DataFrame(index=series.index)
            
            # Compute matrix profile
            mp = stumpy.stump(clean_series, m=self.window_size)
            
            # Matrix profile values (distances)
            mp_values = mp[:, 0]
            
            # Pad to match original length
            padded_mp = np.concatenate([np.full(self.window_size - 1, np.nan), mp_values])
            
            # Threshold for discords (anomalies)
            threshold = np.nanpercentile(padded_mp, self.threshold_percentile)
            anomalies = padded_mp > threshold
            
            # Normalize scores
            normalized = (padded_mp - np.nanmin(padded_mp)) / (np.nanmax(padded_mp) - np.nanmin(padded_mp) + 1e-10)
            
            # Map back to original index
            original_index = series.dropna().index
            result_index = original_index[:len(padded_mp)] if len(padded_mp) <= len(original_index) else original_index
            
            result = pd.DataFrame({
                f'{self.name}_distance': padded_mp[:len(result_index)],
                f'{self.name}_anomaly': anomalies[:len(result_index)].astype(int),
                f'{self.name}_score': normalized[:len(result_index)]
            }, index=result_index)
            
            return result.reindex(series.index)
            
        except ImportError:
            logger.warning("STUMPY not available, skipping Matrix Profile")
            return pd.DataFrame(index=series.index)


class MLAnomalyEnsemble:
    """
    Ensemble of all ML-based anomaly detection methods.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.detectors = self._initialize_detectors()
        
    def _initialize_detectors(self) -> Dict:
        """Initialize all ML detectors."""
        return {
            'isolation_forest': IsolationForestDetector(
                contamination=self.config.get('iso_contamination', 0.05),
                n_estimators=self.config.get('iso_n_estimators', 200)
            ),
            'lof': LOFDetector(
                n_neighbors=self.config.get('lof_n_neighbors', 20),
                contamination=self.config.get('lof_contamination', 0.05)
            ),
            'ocsvm': OneClassSVMDetector(
                kernel=self.config.get('svm_kernel', 'rbf'),
                nu=self.config.get('svm_nu', 0.05)
            ),
            'autoencoder': AutoencoderDetector(
                encoding_dim=self.config.get('ae_encoding_dim', 8),
                epochs=self.config.get('ae_epochs', 50)
            ),
            'dbscan': DBSCANDetector(
                eps=self.config.get('dbscan_eps', 0.5),
                min_samples=self.config.get('dbscan_min_samples', 5)
            ),
            'gmm': GMMDetector(
                n_components=self.config.get('gmm_n_components', 3)
            ),
            'matrix_profile': MatrixProfileDetector(
                window_size=self.config.get('mp_window', 20)
            )
        }
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all ML anomaly detectors to the data.
        """
        results = {}
        
        # Prepare feature set
        feature_cols = [
            'returns', 'log_returns', 'volume_ratio', 'volatility_20d',
            'price_range', 'gap', 'atr_14'
        ]
        feature_cols = [c for c in feature_cols if c in data.columns]
        
        if len(feature_cols) < 2:
            # Fallback to all numeric columns
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        feature_data = data[feature_cols].copy()
        
        logger.info(f"Running ML anomaly detection with {len(feature_cols)} features...")
        
        # Isolation Forest
        results['iso_forest'] = self.detectors['isolation_forest'].fit_transform(feature_data)
        
        # LOF
        results['lof'] = self.detectors['lof'].fit_transform(feature_data)
        
        # One-Class SVM
        results['ocsvm'] = self.detectors['ocsvm'].fit_transform(feature_data)
        
        # Autoencoder
        results['autoencoder'] = self.detectors['autoencoder'].fit_transform(feature_data)
        
        # DBSCAN
        results['dbscan'] = self.detectors['dbscan'].fit_transform(feature_data)
        
        # GMM
        results['gmm'] = self.detectors['gmm'].fit_transform(feature_data)
        
        # Matrix Profile on returns
        if 'returns' in data.columns:
            results['matrix_profile'] = self.detectors['matrix_profile'].fit_transform(data['returns'])
        
        # Combine all results
        combined = pd.concat([df for df in results.values() if not df.empty], axis=1)
        
        # Calculate ensemble anomaly score
        anomaly_cols = [col for col in combined.columns if col.endswith('_anomaly')]
        score_cols = [col for col in combined.columns if col.endswith('_score')]
        
        combined['ml_anomaly_count'] = combined[anomaly_cols].sum(axis=1)
        combined['ml_anomaly_mean_score'] = combined[score_cols].mean(axis=1)
        combined['ml_ensemble_signal'] = (combined['ml_anomaly_count'] >= 3).astype(int)
        
        logger.info(f"ML ensemble complete: {len(combined.columns)} features generated")
        
        return combined


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Fetch RKLB data
    data = yf.Ticker("RKLB").history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['volatility_20d'] = data['returns'].rolling(20).std() * np.sqrt(252)
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['atr_14'] = data['high'].rolling(14).max() - data['low'].rolling(14).min()
    
    # Run ensemble
    ensemble = MLAnomalyEnsemble()
    results = ensemble.fit_transform(data)
    
    print("ML Anomaly Results:")
    print(results.tail(20))
    
    # Show anomaly summary
    anomaly_days = results[results['ml_ensemble_signal'] == 1]
    print(f"\nTotal ML ensemble anomaly signals: {len(anomaly_days)}")
