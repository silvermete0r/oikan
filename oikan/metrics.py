import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

def _ensure_numpy(x):
    """Convert PyTorch tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def evaluate_regression(y_test, y_pred):
    """Evaluate regression performance."""
    y_test = _ensure_numpy(y_test)
    y_pred = _ensure_numpy(y_pred)
    
    mse = np.mean((y_test - y_pred)**2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    
    # Print table
    header = f"+{'-'*23}+{'-'*12}+"
    print(header)
    print(f"| {'Metric':21} | {'Value':9} |")
    print(header)
    print(f"| {'MSE':21} | {mse:9.4f} |")
    print(f"| {'MAE':21} | {mae:9.4f} |")
    print(f"| {'RMSE':21} | {rmse:9.4f} |")
    print(header)
    
    return mse, mae, rmse

def evaluate_classification(y_test, y_pred):
    """Evaluate classification performance."""
    y_test = _ensure_numpy(y_test)
    y_pred = _ensure_numpy(y_pred)
    
    accuracy = np.mean(y_test == y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    h_loss = hamming_loss(y_test, y_pred)
    
    # Print table
    header = f"+{'-'*15}+{'-'*12}+"
    print(header)
    print(f"| {'Metric':13} | {'Value':9} |")
    print(header)
    print(f"| {'Accuracy':13} | {accuracy:9.4f} |")
    print(f"| {'Precision':13} | {precision:9.4f} |")
    print(f"| {'Recall':13} | {recall:9.4f} |")
    print(f"| {'F1-score':13} | {f1:9.4f} |")
    print(f"| {'Hamming Loss':13} | {h_loss:9.4f} |")
    print(header)
    
    return accuracy, precision, recall, f1, h_loss