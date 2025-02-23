import torch
import numpy as np
from oikan.model import OIKAN
from oikan.trainer import train_classification
from oikan.visualize import visualize_classification
from oikan.symbolic import extract_symbolic_formula, test_symbolic_formula, plot_symbolic_formula, extract_latex_formula
from oikan.metrics import evaluate_classification

if __name__ == "__main__":
    # Generate two clusters for binary classification
    n_samples = 1000
    X = np.vstack([
        np.random.randn(n_samples, 2) + np.array([2, 2]),
        np.random.randn(n_samples, 2) + np.array([-2, -2])
    ])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    X_train = torch.FloatTensor(X)
    y_train = torch.LongTensor(y)
    
    # Initialize and train model
    model = OIKAN(input_dim=2, output_dim=2, hidden_units=10)
    train_classification(model, (X_train, y_train), epochs=100)
    
    # Save the trained model
    torch.save(model.state_dict(), "oikan_classification_model.pth")
    
    # Demonstrate reusability: load the saved model
    loaded_model = OIKAN(input_dim=2, output_dim=2, hidden_units=10)
    loaded_model.load_state_dict(torch.load("oikan_classification_model.pth"))
    
    # Use the loaded model for evaluation and visualization
    evaluate_classification(loaded_model, X, y)
    visualize_classification(loaded_model, X, y)
    
    # Extract and display the symbolic formula using the loaded model
    formula = extract_symbolic_formula(loaded_model, X, mode='classification')
    print("Approximate symbolic formula:", formula)
    test_symbolic_formula(loaded_model, X, mode='classification')
    plot_symbolic_formula(loaded_model, X, mode='classification')
    
    # Get LaTeX representation of the symbolic formula
    latex_formula = extract_latex_formula(loaded_model, X, mode='classification')
    print("LaTeX representation of the symbolic formula:", latex_formula)