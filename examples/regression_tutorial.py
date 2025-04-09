import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from oikan.model import OIKANRegressor

dataset = load_diabetes()

X, y = dataset.data, dataset.target
feature_names = dataset.feature_names

# Print dataset info
print("Dataset Overview")
print("-" * 50)
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Number of samples: {X.shape[0]}")

# Preprocess data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Normalize target variable
y_scaler = RobustScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Initialize and train OIKAN regressor
model = OIKANRegressor(
    hidden_dims=[32, 16],  # Smaller architecture
    num_basis=8,          # Fewer basis functions
    degree=2              # Lower degree
)

print("\nTraining OIKAN regressor...")
model.fit(X_train, y_train, epochs=100, lr=0.001)  # Lower learning rate

# Make predictions and inverse transform
y_pred = y_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate R² score
r2 = r2_score(y_test_orig, y_pred)
print(f"\nR² Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.5)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('OIKAN Regression: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

# After the original model prediction, add symbolic prediction evaluation
print("\nEvaluating Symbolic Formula Prediction:")
print("-" * 50)

# Get symbolic predictions
y_pred_symbolic = y_scaler.inverse_transform(
    model.symbolic_predict(X_test).reshape(-1, 1)
).ravel()

# Calculate R² score for symbolic predictions
r2_symbolic = r2_score(y_test_orig, y_pred_symbolic)
print(f"R² Score (Original Model): {r2:.4f}")
print(f"R² Score (Symbolic Formula): {r2_symbolic:.4f}")

# Plot comparison
plt.figure(figsize=(12, 5))

# Original model predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test_orig, y_pred, alpha=0.5, label='Original')
plt.plot([y_test_orig.min(), y_test_orig.max()], 
         [y_test_orig.min(), y_test_orig.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Original Model Predictions')
plt.legend()
plt.grid(True)

# Symbolic formula predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test_orig, y_pred_symbolic, alpha=0.5, label='Symbolic', color='green')
plt.plot([y_test_orig.min(), y_test_orig.max()], 
         [y_test_orig.min(), y_test_orig.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Symbolic Formula Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Analyze prediction differences
mae_diff = np.mean(np.abs(y_pred - y_pred_symbolic))
print(f"\nMean Absolute Difference between Original and Symbolic predictions: {mae_diff:.4f}")

# Extract and display symbolic formula
print("\nExtracted Symbolic Formula:")
print("-" * 50)
formulas = model.get_symbolic_formula()
for i, feature in enumerate(feature_names):
    print(f"{feature}: {formulas[i] if isinstance(formulas, list) else formulas}")
print("-" * 50)

# Print feature importance comparison
print("\nFeature Importance Analysis:")
print("-" * 50)
original_importance = model.get_feature_scores()
symbolic_terms = model.get_symbolic_formula()

for i, (feature, importance) in enumerate(zip(feature_names, original_importance)):
    print(f"\n{feature}:")
    print(f"Original Importance: {importance:.4f}")
    print(f"Symbolic Formula: {symbolic_terms[i]}")

# Feature importance analysis
importance_scores = model.get_feature_scores()

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_names, importance_scores)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('OIKAN Feature Importance Analysis')
plt.tight_layout()
plt.show()
