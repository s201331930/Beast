import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class StatisticalDetector:
    def __init__(self, window=20, threshold=2.5):
        self.window = window
        self.threshold = threshold

    def detect(self, df, column='Log_Return'):
        """
        Detects anomalies using rolling Z-Score.
        """
        roll_mean = df[column].rolling(window=self.window).mean()
        roll_std = df[column].rolling(window=self.window).std()
        z_score = (df[column] - roll_mean) / roll_std
        
        anomalies = np.abs(z_score) > self.threshold
        return anomalies.astype(int)

class IsolationForestDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.feature_cols = []

    def fit_predict(self, df, feature_cols):
        self.feature_cols = feature_cols
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # -1 for outliers, 1 for inliers. Convert to 1 for outliers, 0 for inliers
        preds = self.model.fit_predict(X_scaled)
        return np.where(preds == -1, 1, 0)

class OneClassSVMDetector:
    def __init__(self, nu=0.05, kernel="rbf", gamma=0.1):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.scaler = StandardScaler()

    def fit_predict(self, df, feature_cols):
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        preds = self.model.fit_predict(X_scaled)
        return np.where(preds == -1, 1, 0)

# Deep Learning: LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        # Encoder
        _, (h_n, _) = self.encoder(x)
        # Repeat vector
        h_n = h_n[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        # Decoder
        x_recon, _ = self.decoder(h_n)
        return x_recon

class LSTMAnomalyDetector:
    def __init__(self, sequence_length=10, hidden_dim=32, epochs=50):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = 0

    def create_sequences(self, data):
        xs = []
        for i in range(len(data) - self.sequence_length):
            x = data[i:(i + self.sequence_length)]
            xs.append(x)
        return np.array(xs)

    def fit_predict(self, df, feature_cols):
        data = df[feature_cols].values
        data_scaled = self.scaler.fit_transform(data)
        
        X = self.create_sequences(data_scaled)
        X_tensor = torch.FloatTensor(X)
        
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model = LSTMAutoencoder(input_dim=len(feature_cols), hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        print("Training LSTM Autoencoder...")
        self.model.train()
        for epoch in range(self.epochs):
            loss_val = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss_val/len(dataloader):.4f}")
                
        # Calculate reconstruction error
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            loss = torch.mean((X_tensor - reconstructions)**2, dim=[1, 2])
            
        # Set threshold (e.g., 95th percentile of errors)
        self.threshold = np.percentile(loss.numpy(), 95)
        
        anomalies = (loss.numpy() > self.threshold).astype(int)
        
        # Pad initial values to match df length
        padded_anomalies = np.concatenate([np.zeros(self.sequence_length), anomalies])
        return padded_anomalies

