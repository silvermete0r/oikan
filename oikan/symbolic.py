import torch
import numpy as np

def extract_symbolic_formula_regression(model, X):
    """Simple coefficient-based formula extraction"""
    model.eval()
    with torch.no_grad():
        # Get weights from the first adaptive layer
        weights = model.interpretable_layers[0].weights.numpy()
        # Simplified representation
        terms = []
        for i in range(X.shape[1]):
            coef = np.abs(weights[i]).mean()
            if coef > 0.1:  # threshold for significance
                terms.append(f"{coef:.2f}*x{i+1}")
    
    return " + ".join(terms) if terms else "0"

def extract_symbolic_formula_classification(model, X):
    """Extract classification boundary formula"""
    return extract_symbolic_formula_regression(model, X) + " = 0"
