# ==== STEP 1: Generate synthetic multivariate data (y = sin(x1) + cos(x2) + x3^2 + noise) ====
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from oikan.model import OIKANRegressor

# Generate synthetic data: y = cos(sin(x1) + cos(x2) + x3^2) + noise
np.random.seed(0)
N = 500
x1 = np.random.uniform(0, 2*np.pi, N)
x2 = np.random.uniform(0, 2*np.pi, N)
x3 = np.random.uniform(-1, 1, N)
noise = np.random.normal(0, 0.1, N)
y = np.cos(np.sin(x1) + np.cos(x2) + x3**2) + noise
X = np.column_stack([x1, x2, x3])

# ==== STEP 2: Split data into training and test sets ====
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== STEP 3: Convert data to torch tensors ====
# Convert to torch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1,1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1,1)

# ==== STEP 4: Initialize and train the OIKAN regressor ====
# Initialize and train OIKAN regressor
model = OIKANRegressor()
model.fit(X_train_tensor, y_train_tensor, epochs=200, lr=0.01, verbose=True)

# ==== STEP 5: Obtain neural network predictions ====
# Neural network predictions
preds = model.predict(X_test_tensor)
print("Neural NN Predictions (first five):", preds[:5])

# ==== STEP 6: Obtain symbolic predictions using the extracted formula ====
# Symbolic prediction using extracted formula
symbolic_preds = model.symbolic_predict(X_test)
print("Symbolic Predictions (first five):", symbolic_preds[:5])

# ==== STEP 7: Save and analyze the symbolic formula ====
model.save_symbolic_formula("outputs/multivariate_symbolic_formula.txt")
print("\nExtracted Symbolic Formula:")
formulas = model.get_symbolic_formula()
print("\nFeature-wise formulas:")
for i, formula in enumerate(formulas):
    print(f"Feature {i+1} ({['x1', 'x2', 'x3'][i]}): {formula}")

# ==== STEP 8: Evaluate regression performance ====
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print("\nRegression Performance:")
print(f"{'Metric':<10}{'Value':>10}")
print(f"{'MSE':<10}{mse:>10.4f}")
print(f"{'R²':<10}{r2:>10.4f}")

# ==== STEP 9: Plot the results and training loss history ====
# Plot results
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, label="True", color="blue", alpha=0.6)
plt.scatter(range(len(y_test)), preds, label="Neural Prediction", color="red", marker="x")
plt.scatter(range(len(y_test)), symbolic_preds, label="Symbolic Prediction", color="green", marker="^")
plt.xlabel("Test sample index")
plt.ylabel("Target value")
plt.title("Multivariate Function Approximation with OIKAN")
plt.legend()
plt.show()

# Plot training loss history
loss_history = model.get_loss_history()
plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History (Multivariate)")
plt.grid(True)
plt.show()