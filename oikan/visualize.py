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
    
    if X.shape[1] > 2:
        # SVD projection for high-dimensional inputs.
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        principal = Vt[:2]  # shape: (2, D)
        X_proj = (X - X_mean) @ principal.T
        
        x1, x2 = X_proj[:, 0], X_proj[:, 1]
        x_min, x_max = x1.min() - 1, x1.max() + 1
        y_min, y_max = x2.min() - 1, x2.max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        # Inverse transform grid points to original space.
        X_grid = X_mean + grid_2d @ principal
        
        with torch.no_grad():
            X_grid_tensor = torch.FloatTensor(X_grid)
            Z = model(X_grid_tensor)
            Z = torch.argmax(Z, dim=1).numpy()
            Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, alpha=0.8)
        plt.title("Classification Visualization (SVD Projection)")
        plt.show()
        
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
        x_min, x_max = x1.min() - 1, x1.max() + 1
        y_min, y_max = x2.min() - 1, x2.max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_2d = np.c_[xx.ravel(), yy.ravel()]
        X_grid = grid_2d

        with torch.no_grad():
            X_grid_tensor = torch.FloatTensor(X_grid)
            Z = model(X_grid_tensor)
            Z = torch.argmax(Z, dim=1).numpy()
            Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.title("Classification Visualization")
        plt.show()