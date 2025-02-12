import matplotlib.pyplot as plt
import torch

# Regression Visualization Function
def visualize_regression(model, X, y):
    with torch.no_grad():
        y_pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    plt.scatter(X[:, 0], y, label='True Data')
    plt.scatter(X[:, 0], y_pred, label='OIKAN Predictions', color='r')
    plt.legend()
    plt.show()

# Classification visualization
def visualize_classification(model, X, y):
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32))
        preds = torch.argmax(outputs, dim=1).numpy()
    plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='viridis', edgecolor='k')
    plt.title("Classification Results")
    plt.show()