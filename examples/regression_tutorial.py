# Regression Tutorial using California Housing Dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from oikan import OIKANRegressor

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize OIKANRegressor with verbose=True
model = OIKANRegressor(
    hidden_sizes=[32, 32], 
    activation='relu', 
    augmentation_factor=10,
    alpha=0.1, 
    sigma=0.1, 
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

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Get symbolic formula
formula = model.get_formula()
print("Symbolic Formula:", formula)

# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("outputs/california_housing_model.json")

# Load the model (optional)
print("Loaded Model:")
loaded_model = OIKANRegressor()
loaded_model.load("outputs/california_housing_model.json")
formula_loaded = loaded_model.get_formula(type='original')
print("> Symbolic Formula (loaded):", formula_loaded)

simplified_formula = loaded_model.get_formula(type='sympy')
print("Symbolic Formula (simplified):", simplified_formula)

latex_formula = loaded_model.get_formula(type='latex')
print("LaTeX Formula:", latex_formula)