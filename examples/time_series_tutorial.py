import numpy as np
import torch
import matplotlib.pyplot as plt
from oikan.model import OIKANRegressor

# ==== STEP 1: Generate synthetic time series with a trend (sine wave + linear trend + noise) ====
np.random.seed(0)
T = 300
t = np.arange(T)
series = 0.05 * t + np.sin(0.1 * t) + np.random.normal(0, 0.2, T)

# ==== STEP 2: Create sliding window features for forecasting ====
def create_dataset(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_dataset(series, window_size)

# ==== STEP 3: Split data into training and test sets ====
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==== STEP 4: Convert data to torch tensors ====
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1,1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1,1)

# ==== STEP 5: Initialize and train the OIKAN regressor for forecasting ====
model = OIKANRegressor()
model.fit(X_train_tensor, y_train_tensor, epochs=250, lr=0.005, verbose=True)

# ==== STEP 6: Obtain neural network and symbolic predictions ====
preds = model.predict(X_test_tensor)
print("Forecast Predictions (first five):", preds[:5])
symbolic_preds = model.symbolic_predict(X_test)
print("Symbolic Forecast Predictions (first five):", symbolic_preds[:5])

# ==== STEP 7: Plot actual vs. predicted values for the time series forecast ====
plt.figure(figsize=(8,5))
plt.plot(range(len(y_test)), y_test, label="Actual", color="blue")
plt.plot(range(len(y_test)), preds, label="NN Forecast", color="red", linestyle="--")
plt.plot(range(len(y_test)), symbolic_preds, label="Symbolic Forecast", color="green", linestyle=":")
plt.xlabel("Time step")
plt.ylabel("Series value")
plt.title("Time Series Forecasting with OIKAN")
plt.legend()
plt.show()

# ==== STEP 8: Plot the training loss history ====
loss_history = model.get_loss_history()
plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History (Time Series Forecasting)")
plt.grid(True)
plt.show()
