import numpy as np
import torch

def evaluate_regression(model, X, y):
    '''Evaluate regression performance by computing MSE, MAE, and RMSE.'''
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X)).numpy().ravel()
    mse = np.mean((y - y_pred)**2)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(mse)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    return mse, mae, rmse

def evaluate_classification(model, X, y):
    '''Evaluate classification accuracy by comparing model predictions and true labels.'''
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        y_pred = torch.argmax(logits, dim=1).numpy()
    accuracy = np.mean(y_pred == y)
    print("Classification Accuracy:", accuracy)
    return accuracy