import torch
import numpy as np
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.visualize import visualize_regression
from oikan.symbolic import extract_symbolic_formula_regression

# Example Usage
if __name__ == "__main__":
    X1 = np.linspace(-1, 1, 100).reshape(-1, 1)
    X2 = np.linspace(-1, 1, 100).reshape(-1, 1)
    X = np.hstack((X1, X2))
    y = np.sin(3 * X1) + np.cos(2 * X2) + 0.1 * np.random.randn(100, 1)
    
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    train_loader = (X_train, y_train)
    
    model = OIKAN(input_dim=2, output_dim=1)
    train(model, train_loader)
    visualize_regression(model, X, y)
    
    print("Extracted Symbolic Formula:", extract_symbolic_formula_regression(model, X))