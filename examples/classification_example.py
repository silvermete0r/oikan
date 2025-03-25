import numpy as np
from oikan.model import OIKAN
from oikan.visualize import visualize_classification
from oikan.metrics import evaluate_classification
from sklearn.datasets import make_moons
    
if __name__ == "__main__":
    # Generate two moons dataset
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    
    # Create OIKAN model with classification mode
    model = OIKAN(input_dim=2, output_dim=1, mode='classification')
    
    # Train the model
    model.fit(X, y, epochs=100, verbose=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate the model
    evaluate_classification(y, y_pred)
    
    # Visualize decision boundaries
    visualize_classification(X, y, y_pred)
    
    # Extract and print symbolic formula
    print('\nSymbolic Decision Boundary:')
    print(model.extract_symbolic_formula(X))
    
    print('\nLatex Formula:')
    print(model.extract_latex_formula(X))
    
    # Plot symbolic formula representation
    model.plot_symbolic_formula(X)
    
    # Test symbolic formula accuracy
    print('\nTesting symbolic formula approximation:')
    model.test_symbolic_formula(X)