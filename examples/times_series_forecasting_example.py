import torch
import numpy as np
import matplotlib.pyplot as plt
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.metrics import evaluate_regression
from oikan.visualize import visualize_time_series_forecasting

def main():
    # Generate synthetic time series (sine wave with noise)
    t = np.linspace(0, 20 * np.pi, 2000)
    series = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    # Create sliding window dataset: window_size as input, next value as target
    window_size = 20
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)  # shape: (samples, window_size)
    y = np.array(y)  # shape: (samples,)
    
    # Reshape for forecasting: add feature dim so that input shape is [batch, seq_len, features]
    X_train = torch.FloatTensor(X).unsqueeze(-1)  # shape: (samples, window_size, 1)
    y_train = torch.FloatTensor(y).unsqueeze(1)     # shape: (samples, 1)
    
    # Initialize model in forecast_mode (time series forecasting)
    model = OIKAN(input_dim=1, output_dim=1, hidden_units=10, basis_type='combo', forecast_mode=True)
    
    # Train the forecasting model
    train(model, (X_train, y_train), epochs=100, lr=0.001)
    
    # Evaluate forecasting performance
    evaluate_regression(model, X_train, y_train.numpy())
    
    # Visualize time series forecasting using the new function
    visualize_time_series_forecasting(model, X_train, y_train)

if __name__ == '__main__':
    main()