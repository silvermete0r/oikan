import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from oikan.model import OIKAN
import torch

# Load Wine dataset
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

print("Dataset Features:", feature_names)
print("Number of features:", len(feature_names))
print("Number of classes:", len(np.unique(y)))
print("Number of samples:", X.shape[0])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Create and train OIKAN model with history tracking
model = OIKAN(mode='classification')

# Convert data to correct tensor types
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # Use LongTensor for classification labels
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

model, history = model.fit(X_train, y_train, epochs=150, lr=0.005, verbose=True, history=True)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['epoch'], history['loss'])
plt.title('OIKAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Make predictions and convert to numpy for evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_test_np = y_test.cpu().numpy()  # Convert to numpy for metrics

# Evaluate model
from oikan.metrics import evaluate_classification
accuracy, precision, recall, f1, h_loss = evaluate_classification(y_test_np, y_pred)

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add value annotations
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Extract and display symbolic formula
print('\nSymbolic Decision Boundary:')
formula = model.extract_symbolic_formula(X_test)
print(formula)

print('\nLatex Formula:')
latex = model.extract_latex_formula(X_test)
print(latex)

# Test symbolic formula accuracy
print("\nTesting Symbolic Formula Accuracy:")
model.test_symbolic_formula(X_test)

# Visualize formula structure (if not too complex)
try:
    model.plot_symbolic_formula(X_test)
except Exception as e:
    print("\nFormula visualization skipped:", str(e))