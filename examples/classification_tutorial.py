from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from oikan import OIKANClassifier

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Standard Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize OIKANClassifier with verbose=True
model = OIKANClassifier(
    hidden_sizes=[32, 32], 
    activation='relu',
    augmentation_factor=5,
    alpha=0.1,
    l1_ratio=0.5,
    sigma=3,
    epochs=100,
    lr=0.001,
    batch_size=32, 
    top_k=10,
    evaluate_nn=True,
    verbose=True,
    random_state=42
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Get symbolic formulas for each class
formulas = model.get_formula()
for i, formula in enumerate(formulas):
    print(f"Class {i} Formula:", formula)

# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("../outputs/iris_model.json")

# Load the model (optional)
print("Loaded Model:")
loaded_model = OIKANClassifier()
loaded_model.load("../outputs/iris_model.json")
formulas_loaded = loaded_model.get_formula(type='original')
for formula in formulas_loaded:
    print(formula)

print("Simplified Formulas:")
simplified_formulas = loaded_model.get_formula(type='sympy')
for simplified_formula in simplified_formulas:
    print(simplified_formula)

print("LaTeX Formulas:")
latex_formulas = loaded_model.get_formula(type='latex')
for latex_formula in latex_formulas:
    print(latex_formula)