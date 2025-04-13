# ==== STEP 1: Load example classification data (Iris dataset) ====
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from oikan.model import OIKANClassifier

iris = load_iris()
X = iris.data            # shape (150, 4)
y = iris.target          # shape (150,)
print("Iris dataset loaded. Classes:", iris.target_names)

# ==== STEP 2: Split into train and test sets ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ==== STEP 3: Convert data to torch tensors ====
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# ==== STEP 4: Initialize and train the OIKAN classifier ====
model = OIKANClassifier()
model.fit(X_train_tensor, y_train_tensor, epochs=100, lr=0.01, verbose=True)  

# ==== STEP 5: Get predicted probabilities and classes from the neural network ====
probas = model.predict_proba(X_test_tensor)
preds = model.predict(X_test_tensor)
print("Neural Network Predicted Probabilities (first five):")
print(probas[:5])
print("Predicted Classes (first five):")
print(preds[:5])

# ==== STEP 6: Evaluate performance ====
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# ==== STEP 7: Extract and display the symbolic formula ====
symbolic_formula = model.get_symbolic_formula()
print("Extracted Symbolic Formula:")
print(symbolic_formula)

# ==== STEP 8: Obtain symbolic prediction probabilities ====
symbolic_probas = model.symbolic_predict_proba(X_test)
print("Symbolic Predicted Probabilities (first five):")
print(symbolic_probas[:5])

# ==== STEP 9: Save symbolic formula and model for later use ====
model.save_symbolic_formula(filename="outputs/classification_symbolic_formula.txt")
model.save_model(filepath="models/classification_model.pth")
if hasattr(model, "get_feature_scores"):
    scores = model.get_feature_scores()
    print("Feature Importance Scores:", scores)

# ==== STEP 10: Compile the symbolic formula into a runnable function and test it ====
compiled_fn = model.compile_symbolic_formula(filename="outputs/classification_symbolic_formula.txt")
sample_input = np.array([X_test[0]])  # test with first sample from test set
print("Compiled symbolic function output for first test sample:", compiled_fn(sample_input))

# ==== STEP 11: Demonstrate model reloading ====
loaded_model = OIKANClassifier()
output_dim = 3  # Iris dataset has 3 classes
loaded_model.model = model._build_network(X_train_tensor.shape[1], output_dim)
loaded_model.load_model(filepath="models/classification_model.pth")
loaded_probas = loaded_model.predict_proba(X_test_tensor)
print("Loaded Model Predicted Probabilities (first five):")
print(loaded_probas[:5])

# ==== STEP 12: Visualize decision boundaries using matplotlib ====

# For 2D visualization, use two features (e.g., features at index 0 and 2)
feature_idx = [0, 2]
X_vis = X_test[:, feature_idx]
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Create full feature input by filling non-selected features with their mean value
X_full = np.zeros((grid.shape[0], X.shape[1]))
for i in range(X.shape[1]):
    if i in feature_idx:
        X_full[:, i] = grid[:, feature_idx.index(i)]
    else:
        X_full[:, i] = np.mean(X[:, i])
grid_tensor = torch.FloatTensor(X_full)
grid_probas = model.predict_proba(grid_tensor)
pred_grid = np.argmax(grid_probas, axis=1).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, pred_grid, alpha=0.3, cmap=ListedColormap(("blue", "red", "green")))
plt.scatter(X_test[:, feature_idx[0]], X_test[:, feature_idx[1]], c=y_test, edgecolor="k", cmap=ListedColormap(("blue", "red", "green")))
plt.title("OIKAN Classification on Iris Dataset")
plt.xlabel("Feature: " + iris.feature_names[feature_idx[0]])
plt.ylabel("Feature: " + iris.feature_names[feature_idx[1]])
plt.show()

# New Section: Plot Training Loss History
import matplotlib.pyplot as plt
loss_history = model.get_loss_history()
plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History (Classification)")
plt.grid(True)
plt.show()