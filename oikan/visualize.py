import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_regression(X, y_test, y_pred, confidence=None):
    """Visualize regression results with confidence intervals."""
    plt.figure(figsize=(10, 6))
    
    if X.shape[1] > 2:
        pca = PCA(n_components=1)
        x_plot = pca.fit_transform(X).flatten()
        plt.xlabel(f'Principal Component (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})')
    else:
        x_plot = X[:, 0]
        plt.xlabel('Input Feature')
    
    # Sort by x for better visualization
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]
    y_test = y_test[sort_idx]
    y_pred = y_pred[sort_idx]
    
    plt.scatter(x_plot, y_test, color='blue', alpha=0.6, label='True Values')
    plt.scatter(x_plot, y_pred, color='red', alpha=0.6, label='Predictions')
    
    if confidence is not None:
        confidence = confidence[sort_idx]
        plt.fill_between(x_plot, y_pred - confidence, y_pred + confidence,
                        color='red', alpha=0.2, label='95% Confidence')
    
    plt.ylabel('Target Variable')
    plt.title('Regression Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_classification(X, y_test, y_pred, confidence=None, model=None):
    """Visualize classification results with decision boundaries.
       If a model is provided, its predictions on a mesh grid are used to plot the boundary.
    """
    # Convert inputs to numpy arrays if needed
    X = np.asarray(X)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    plt.figure(figsize=(10, 6))
    
    # Handle 1D or higher dimensional data
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
    else:
        X_vis = X
        x_label = 'Feature 1'
        y_label = 'Feature 2' if X.shape[1] > 1 else 'Feature 1'
    
    if X_vis.shape[1] > 1:
        margin = 0.5
        x_min, x_max = X_vis[:, 0].min() - margin, X_vis[:, 0].max() + margin
        y_min, y_max = X_vis[:, 1].min() - margin, X_vis[:, 1].max() + margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        if model is not None:
            # Use the provided model to predict over the grid.
            # If PCA was applied, we need to map grid points back to original space.
            if X.shape[1] > 2:
                # Inverse transform is not available; warn user.
                print("Warning: PCA was applied. Provide model that accepts PCA-transformed inputs.")
                grid_points = np.c_[xx.ravel(), yy.ravel()]
            else:
                grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(grid_points).reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        else:
            # Fallback: if y_pred is provided in mesh format.
            if y_pred.size == xx.size:
                Z = y_pred.reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # Plot data points
    scatter = plt.scatter(X_vis[:, 0],
                          X_vis[:, 1] if X_vis.shape[1] > 1 else np.zeros_like(X_vis[:, 0]),
                          c=y_test, cmap='RdYlBu', alpha=0.8, edgecolors='black')
    
    plt.colorbar(scatter, label='Class')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Classification Decision Boundaries')
    plt.grid(True, alpha=0.3)
    plt.show()