import torch
import numpy as np
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.visualize import visualize_regression
from oikan.symbolic import extract_symbolic_formula, test_symbolic_formula, plot_symbolic_formula, extract_latex_formula
from oikan.metrics import evaluate_regression

if __name__ == "__main__":
    # Generate simple 1D data for demonstration
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(1000, 1)
    
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)
    
    # Initialize and train model
    model = OIKAN(input_dim=1, output_dim=1, hidden_units=10)
    train(model, (X_train, y_train), epochs=100)
    
    # Save the trained model
    torch.save(model.state_dict(), "models/oikan_regression_model.pth")
    
    # Demonstrate reusability: load the saved model
    loaded_model = OIKAN(input_dim=1, output_dim=1, hidden_units=10)
    loaded_model.load_state_dict(torch.load("models/oikan_regression_model.pth"))
    
    # Use the loaded model for evaluation and visualization
    evaluate_regression(loaded_model, X, y)
    visualize_regression(loaded_model, X, y)
    
    # Extract and display the symbolic formula using the loaded model
    formula = extract_symbolic_formula(loaded_model, X, mode='regression')
    print("Approximate symbolic formula:", formula)
    test_symbolic_formula(loaded_model, X, mode='regression')
    plot_symbolic_formula(loaded_model, X, mode='regression')
    
    # Get LaTeX representation of the symbolic formula
    latex_formula = extract_latex_formula(loaded_model, X, mode='regression')
    print("LaTeX representation of the symbolic formula:", latex_formula)
