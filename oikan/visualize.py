import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_regression(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        y_pred = model(X_tensor).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color='blue', label='True')
    plt.scatter(X[:, 0], y_pred, color='red', label='Predicted')
    plt.legend()
    plt.show()

def visualize_classification(model, X, y):
    model.eval()
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Make predictions
    with torch.no_grad():
        X_grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(X_grid)
        Z = torch.argmax(Z, dim=1).numpy()
        Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()