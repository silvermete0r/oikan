import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_regression(model, X, y):
    '''Visualize regression results using true vs predicted scatter plots.'''
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X)).numpy()


    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color='blue', label='True')
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