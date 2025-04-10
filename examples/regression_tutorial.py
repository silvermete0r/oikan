import numpy as np
import torch
from oikan.model import OIKANRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==== STEP 1: Load example regression data (Diabetes dataset) ====
data = load_diabetes()
X = data.data            # shape (442, 10)
y = data.target          # shape (442,)
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", data.feature_names)

# ==== STEP 2: Split into train and test sets ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ==== STEP 3: Convert data to torch tensors ====
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# ==== STEP 4: Initialize and train the OIKAN regressor ====
model = OIKANRegressor(hidden_dims=[16, 8], num_basis=10, degree=3, dropout=0.1)
model.fit(X_train_tensor, y_train_tensor, epochs=200, lr=0.01, batch_size=16, verbose=True)  # changed epochs to 200

# ==== STEP 5: Obtain neural network predictions on test set ====
preds = model.predict(X_test_tensor)
print("Neural Network Predictions (first five):")
print(preds[:5])

# ==== STEP 6: Evaluate performance using MSE and R² ====
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}")

# ==== STEP 7: Extract and display the production-ready symbolic formula ====
symbolic_formula = model.get_symbolic_formula()
print("Extracted Symbolic Formula:")
print(symbolic_formula)

# ==== STEP 8: Perform symbolic prediction using the extracted formula ====
symbolic_preds = model.symbolic_predict(X_test)
print("Symbolic Predictions (first five):")
print(symbolic_preds[:5])

# ==== STEP 9: Save symbolic formula and model for later use ====
model.save_symbolic_formula(filename="outputs/regression_symbolic_formula.txt")
model.save_model(filepath="models/regression_model.pth")
scores = model.get_feature_scores()
print("Feature Importance Scores:", scores)

# ==== STEP 10: Compile the symbolic formula into a runnable function and test it ====
compiled_fn = model.compile_symbolic_formula(filename="outputs/regression_symbolic_formula.txt")
sample_input = np.array([X_test[0]])  # test with first sample from test set
print("Compiled symbolic function output for first test sample:", compiled_fn(sample_input))

# ==== STEP 11: Demonstrate model reloading ====
loaded_model = OIKANRegressor(hidden_dims=[16, 8], num_basis=10, degree=3, dropout=0.1)
loaded_model.model = model._build_network(X_train_tensor.shape[1], y_train_tensor.shape[1])
loaded_model.load_model(filepath="models/regression_model.pth")
loaded_preds = loaded_model.predict(X_test_tensor)
print("Loaded Model Predictions (first five):")
print(loaded_preds[:5])

# ==== STEP 12: Visualize test performance with matplotlib ====
plt.figure(figsize=(8,5))
plt.scatter(range(len(y_test)), y_test, label="True Target", color="blue", alpha=0.6)
plt.scatter(range(len(y_test)), preds, label="Neural Prediction", color="red")
plt.scatter(range(len(y_test)), symbolic_preds, label="Symbolic Prediction", color="green", marker="x")
plt.title("OIKAN Regression on Diabetes Dataset")
plt.xlabel("Test Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.show()

# New Section: Plot Training Loss History
import matplotlib.pyplot as plt
loss_history = model.get_loss_history()
plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History (Regression)")
plt.grid(True)
plt.show()