import torch
import numpy as np
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.visualize import visualize_regression, visualize_regression_3d
from oikan.symbolic import extract_symbolic_formula, test_symbolic_formula, plot_symbolic_formula, extract_latex_formula

if __name__ == "__main__":
    # Generate simple 1D data for demonstration
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(100, 1)
    
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)
    
    # Initialize and train model
    model = OIKAN(input_dim=1, output_dim=1, hidden_units=10)
    train(model, (X_train, y_train), epochs=100)
    
    # Visualize results
    visualize_regression(model, X, y)
    
    # Extract and print symbolic formula
    formula = extract_symbolic_formula(model, X, mode='regression')
    print("Approximate symbolic formula:", formula)
    test_symbolic_formula(model, X, mode='regression')

    # Plot symbolic formula
    plot_symbolic_formula(model, X, mode='regression')

    # Get LaTeX representation of the symbolic formula
    latex_formula = extract_latex_formula(model, X, mode='regression')
    print("LaTeX representation of the symbolic formula:", latex_formula)