import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_regression(X, y_test, y_pred):
    '''Visualize regression results using true vs predicted scatter plots.'''
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y_test, color='blue', label='True')
    plt.scatter(X[:, 0], y_pred, color='red', label='Predicted')
    plt.legend()
    plt.show()

def visualize_classification(model, X, y):
    '''Visualize classification decision boundaries. For high-dimensional data, uses SVD projection.'''
    model.eval()
    if X.shape[1] > 2:
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        principal = Vt[:2]
        X_proj = (X - X_mean) @ principal.T
        x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
        y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        X_grid = X_mean + grid_2d @ principal
        with torch.no_grad():
            Z = model(torch.FloatTensor(X_grid))
            Z = torch.argmax(Z, dim=1).numpy().reshape(xx.shape)
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, alpha=0.8)
        plt.title("Classification Visualization (SVD Projection)")
        plt.show()
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        with torch.no_grad():
            Z = model(torch.FloatTensor(grid_2d))
            Z = torch.argmax(Z, dim=1).numpy().reshape(xx.shape)
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)

def visualize_time_series_forecasting(model, X, y):
    '''
    Visualize time series forecasting results by plotting true vs predicted values.
    Expected X shape: [samples, seq_len, features] and y: true targets.
    '''
    model.eval()
    with torch.no_grad():
        y_pred = model(X).detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(y, label='True', marker='o', linestyle='-')
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Time Series Forecasting Visualization")
    plt.legend()
    plt.show()