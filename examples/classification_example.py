import torch
import numpy as np
import matplotlib.pyplot as plt
from oikan.model import OIKAN
from oikan.trainer import train_classification
from oikan.visualize import visualize_classification
from oikan.symbolic import extract_symbolic_formula_classification

# Generate synthetic 2D classification data
np.random.seed(42)
n_samples = 100
X_class = np.vstack((
    np.random.randn(n_samples, 2) + np.array([2, 2]),
    np.random.randn(n_samples, 2) + np.array([-2, -2])
))
y_class = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(np.int64)

X_train = torch.tensor(X_class, dtype=torch.float32)
y_train = torch.tensor(y_class, dtype=torch.long)
train_loader = (X_train, y_train)

# Initialize and train OIKAN for classification (using 2 output neurons)
model = OIKAN(input_dim=2, output_dim=2)
train_classification(model, train_loader, epochs=100, lr=0.01)

# Visualize classification results and print symbolic decision boundary
visualize_classification(model, X_class, y_class)
symbolic_formula = extract_symbolic_formula_classification(model, X_train.numpy())
print("Extracted Symbolic Decision Boundary:", symbolic_formula)
