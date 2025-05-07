from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from oikan import OIKANClassifier

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize OIKANClassifier with verbose=True
model = OIKANClassifier(
    hidden_sizes=[32, 32], 
    activation='relu', 
    augmentation_factor=10, 
    polynomial_degree=2, 
    alpha=0.1, 
    sigma=0.1, 
    epochs=100, 
    lr=0.001, 
    batch_size=32, 
    verbose=True
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Get symbolic formulas for each class
formulas = model.get_formula()
for i, formula in enumerate(formulas):
    print(f"Class {i} Formula:", formula)

# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("iris_model.txt")