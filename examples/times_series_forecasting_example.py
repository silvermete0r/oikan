import torch
import numpy as np
import matplotlib.pyplot as plt
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.metrics import evaluate_regression
from oikan.visualize import visualize_time_series_forecasting

def main():
    # Generate synthetic time series data
    t = np.linspace(0, 20 * np.pi, 2000)
    series = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    # Create sliding window dataset
    window_size = 20
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for forecasting: [batch, seq_len, features]
    X_train = torch.FloatTensor(X).unsqueeze(-1)
    y_train = torch.FloatTensor(y).unsqueeze(1)
    
    # Initialize model in forecast mode
    model = OIKAN(input_dim=1, output_dim=1, hidden_units=10, basis_type='bsplines', forecast_mode=True)
    
    # Train and evaluate the model
    train(model, (X_train, y_train), epochs=100, lr=0.001)
    evaluate_regression(model, X_train, y_train.numpy())
    visualize_time_series_forecasting(model, X_train, y_train)

if __name__ == '__main__':
    main()