import torch
import numpy as np
from oikan.model import OIKAN
from oikan.trainer import train_classification
from oikan.visualize import visualize_classification
from oikan.symbolic import extract_symbolic_formula_classification

if __name__ == "__main__":
    # Generate two clusters for binary classification
    n_samples = 100
    X = np.vstack([
        np.random.randn(n_samples, 2) + np.array([2, 2]),
        np.random.randn(n_samples, 2) + np.array([-2, -2])
    ])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    X_train = torch.FloatTensor(X)
    y_train = torch.LongTensor(y)
    
    # Initialize and train model
    model = OIKAN(input_dim=2, output_dim=2, hidden_units=10)
    train_classification(model, (X_train, y_train), epochs=100)
    
    # Visualize results
    visualize_classification(model, X, y)
    
    # Extract and print decision boundary
    formula = extract_symbolic_formula_classification(model, X)
    print("Decision boundary:", formula)
