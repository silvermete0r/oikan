import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from oikan.model import OIKAN
from oikan.visualize import visualize_classification
from oikan.metrics import evaluate_classification

if __name__ == "__main__":
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print("Dataset Features:", feature_names)
    print("Number of features:", len(feature_names))
    print("Number of samples:", X.shape[0])
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create and train OIKAN model
    model = OIKAN(mode='classification')
    model.fit(X_train, y_train, epochs=150, lr=0.005, verbose=True)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("\nTest Set Evaluation:")
    evaluate_classification(y_test, y_pred)
    
    # Visualize decision boundaries (using first two principal components)
    visualize_classification(X_test, y_test, y_pred)
    
    # Extract and display symbolic formulas
    print('\nSymbolic Decision Boundary:')
    print(model.extract_symbolic_formula(X_test))
    
    print('\nLatex Formula:')
    print(model.extract_latex_formula(X_test))
    
    # Test and visualize symbolic formula
    print("\nSymbolic Formula Testing:")
    model.test_symbolic_formula(X_test)
    model.plot_symbolic_formula(X_test)