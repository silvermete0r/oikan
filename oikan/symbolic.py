import torch
from sympy import symbols, simplify, Add

# Regression symbolic extraction
def extract_symbolic_formula_regression(model, input_data):
    symbolic_vars = symbols([f'x{i}' for i in range(input_data.shape[1])])
    
    with torch.no_grad():
        weights = model.mlp[0].weight.cpu().numpy()
        if weights.size == 0:
            print("Warning: Extracted weights are empty.")
            return "NaN"

    formula = sum(weights[0, i] * symbolic_vars[i] for i in range(len(symbolic_vars)))
    return simplify(formula)

# Classification symbolic extraction
def extract_symbolic_formula_classification(model, input_data):
    """
    Extracts a symbolic decision boundary for a two-class classifier.
    Approximates:
      decision = (w[0] - w[1]) · x + (b[0] - b[1])
    where w and b are from the model's final linear layer.
    """
    symbolic_vars = symbols([f'x{i}' for i in range(input_data.shape[1])])
    with torch.no_grad():
        final_layer = model.mlp[-1]
        w = final_layer.weight.cpu().numpy()
        b = final_layer.bias.cpu().numpy()
        if w.shape[0] < 2:
            print("Classification symbolic extraction requires at least 2 classes.")
            return "NaN"
        w_diff = w[0] - w[1]
        b_diff = b[0] - b[1]
    formula = sum(w_diff[i] * symbolic_vars[i] for i in range(len(symbolic_vars))) + b_diff
    return simplify(formula)
