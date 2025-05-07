# Regression Tutorial using California Housing Dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from oikan import OIKANRegressor

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize OIKANRegressor with verbose=True
model = OIKANRegressor(
    hidden_sizes=[64, 64], 
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

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Get symbolic formula
formula = model.get_formula()
print("Symbolic Formula:", formula)

# Get feature importances
importances = model.feature_importances()
print("Feature Importances:", importances)

# Save the model (optional)
model.save("california_housing_model.txt")