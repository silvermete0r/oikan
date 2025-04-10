import numpy as np
import torch
from oikan.model import OIKANRegressor

# Generate synthetic regression data
np.random.seed(0)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 * X.squeeze() + np.random.normal(0, 1, 100)

# Convert data to torch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Initialize and train OIKAN regressor
model = OIKANRegressor(hidden_dims=[16, 8], num_basis=10, degree=3, dropout=0.1)
model.fit(X_tensor, y_tensor, epochs=50, lr=0.01, batch_size=16, verbose=True)

# Inference using neural network predictions
preds = model.predict(X_tensor)
print("Neural Network Predictions:")
print(preds[:5])

# Extract and display production-ready symbolic formula
symbolic_formula = model.get_symbolic_formula()
print("Extracted Symbolic Formula:")
print(symbolic_formula)

# Inference using the symbolic formula approximation
symbolic_preds = model.symbolic_predict(X)
print("Symbolic Predictions:")
print(symbolic_preds[:5])

# Save symbolic formula for production use
model.save_symbolic_formula(filename="outputs/regression_symbolic_formula.txt")