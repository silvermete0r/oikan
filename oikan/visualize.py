import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_regression(model, X, y):
    '''Visualize regression results via a scatter plot, comparing true vs predicted values.'''
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X)).numpy()
    plt.figure(figsize=(10, 6))
    # Plot true values vs predictions
    plt.scatter(X[:, 0], y, color='blue', label='True')
    plt.scatter(X[:, 0], y_pred, color='red', label='Predicted')
    plt.legend()
    plt.title("Regression: True vs Predicted")
    plt.xlabel("Feature 1")
    plt.ylabel("Output")
    plt.show()

def visualize_classification(model, X, y):
    '''Visualize classification decision boundaries for 2D input data.'''
    model.eval()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        # Compute prediction for each point in the grid
        Z = model(torch.FloatTensor(grid_2d))
        Z = torch.argmax(Z, dim=1).numpy().reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    # Draw decision boundaries and scatter the data
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title("Classification Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()