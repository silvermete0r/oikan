import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from oikan.model import OIKAN

# Load and prepare Diabetes dataset (has only 10 features)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

print("Dataset Features:", feature_names)
print("Target Variable: Disease Progression")
print("Number of features:", len(feature_names))
print("Number of samples:", X.shape[0])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create and train OIKAN model with history tracking
model = OIKAN(mode='regression')
model, history = model.fit(X_train, y_train, epochs=200, lr=0.01, verbose=True, history=True)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['loss'])
plt.title('OIKAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
from oikan.metrics import evaluate_regression
mse, mae, rmse = evaluate_regression(y_test, y_pred)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.title('OIKAN: Actual vs Predicted Disease Progression')
plt.tight_layout()
plt.show()

# Extract and display symbolic formula
print('\nSymbolic Formula:')
formula = model.extract_symbolic_formula(X_test)
print(formula)

print('\nLatex Formula:')
latex = model.extract_latex_formula(X_test)
print(latex)

# Test symbolic formula accuracy
print("\nTesting Symbolic Formula Accuracy:")
model.test_symbolic_formula(X_test)

model.plot_symbolic_formula(X_test)