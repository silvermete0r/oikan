import numpy as np
import torch
from oikan.model import OIKANClassifier

# Generate synthetic binary classification data
np.random.seed(0)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear decision boundary

# Convert data to torch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Initialize and train OIKAN classifier
model = OIKANClassifier(hidden_dims=[16, 8], num_basis=10, degree=3, dropout=0.1)
model.fit(X_tensor, y_tensor, epochs=50, lr=0.01, batch_size=16, verbose=True)

# Inference using neural network outputs
probas = model.predict_proba(X_tensor)
preds = model.predict(X_tensor)
print("Neural Network Class Probabilities (first five):")
print(probas[:5])
print("Predicted Classes:")
print(preds[:5])

# Extract and display production-ready symbolic formula
symbolic_formula = model.get_symbolic_formula()
print("Extracted Symbolic Formula:")
print(symbolic_formula)

# Inference using the symbolic formula approximation
symbolic_probas = model.symbolic_predict_proba(X)
print("Symbolic Predicted Probabilities (first five):")
print(symbolic_probas[:5])

# Save symbolic formula for production use
model.save_symbolic_formula(filename="outputs/classification_symbolic_formula.txt")