import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from oikan.model import OIKANClassifier

# Load and prepare dataset
dataset = load_iris()  # Can also use load_breast_cancer() for binary classification
X, y = dataset.data, dataset.target
feature_names = dataset.feature_names
target_names = dataset.target_names

# Print dataset info
print("Dataset Overview")
print("-" * 50)
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Number of classes: {len(target_names)}")
print(f"Class names: {target_names}")
print(f"Number of samples: {X.shape[0]}")
print("-" * 50)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize and train OIKAN classifier
model = OIKANClassifier(
    hidden_dims=[32, 16],  # Smaller architecture for classification
    num_basis=8,           # Number of basis functions
    degree=3               # Degree of B-spline basis
)

print("\nTraining OIKAN classifier...")
model.fit(X_train, y_train, epochs=100, lr=0.01)

# Make predictions with both original model and symbolic formulas
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_pred_symbolic = model.symbolic_predict(X_test)
y_proba_symbolic = model.symbolic_predict_proba(X_test)

# Print classification reports for both methods
print("\nOriginal Model Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nSymbolic Formula Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred_symbolic, target_names=target_names))

# Compare confusion matrices side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names, ax=ax1)
ax1.set_title('Original Model Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

cm_symbolic = confusion_matrix(y_test, y_pred_symbolic)
sns.heatmap(cm_symbolic, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names, ax=ax2)
ax2.set_title('Symbolic Formula Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

plt.tight_layout()
plt.show()

# Compare decision boundaries (for first two features)
if len(target_names) > 2:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(target_names):
        # Original model probabilities
        scatter1 = axes[0, i].scatter(X_test[:, 0], X_test[:, 1], 
                                    c=y_proba[:, i], cmap='viridis',
                                    alpha=0.6)
        plt.colorbar(scatter1, ax=axes[0, i], label=f'P({target})')
        axes[0, i].set_xlabel(feature_names[0])
        axes[0, i].set_ylabel(feature_names[1])
        axes[0, i].set_title(f'Original: P({target})')
        
        # Symbolic formula probabilities
        scatter2 = axes[1, i].scatter(X_test[:, 0], X_test[:, 1], 
                                    c=y_proba_symbolic[:, i], cmap='viridis',
                                    alpha=0.6)
        plt.colorbar(scatter2, ax=axes[1, i], label=f'P({target})')
        axes[1, i].set_xlabel(feature_names[0])
        axes[1, i].set_ylabel(feature_names[1])
        axes[1, i].set_title(f'Symbolic: P({target})')
    
    plt.tight_layout()
    plt.show()

# Analyze prediction differences
prediction_match = np.mean(y_pred == y_pred_symbolic)
print(f"\nPrediction Match Rate between Original and Symbolic: {prediction_match:.4f}")

# Extract and display symbolic formulas for decision boundaries
print("\nExtracted Decision Boundary Formulas:")
print("-" * 50)
formulas = model.get_symbolic_formula()
for i, feature in enumerate(feature_names):
    print(f"{feature}:")
    for j, target in enumerate(target_names):
        formula = formulas[i][j] if isinstance(formulas[i], list) else formulas[i]
        print(f"  -> {target}: {formula}")
print("-" * 50)

# Plot decision probabilities for first two features (if multi-class)
if len(target_names) > 2:
    plt.figure(figsize=(12, 4))
    for i, target in enumerate(target_names):
        plt.subplot(1, 3, i+1)
        scatter = plt.scatter(X_test[:, 0], X_test[:, 1], 
                            c=y_proba[:, i], cmap='viridis',
                            alpha=0.6)
        plt.colorbar(scatter, label=f'P({target})')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'Probability of Class: {target}')
    plt.tight_layout()
    plt.show()

# Feature importance analysis
importance_scores = model.get_feature_scores()

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importance_scores)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('OIKAN Feature Importance Analysis')
plt.tight_layout()
plt.show()
