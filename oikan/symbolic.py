import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functools
import re

ADVANCED_LIB = {
    'x':    lambda x: x,
    'x^2':  lambda x: x**2,
    'exp':  lambda x: np.exp(x),
    'sin':  lambda x: np.sin(x),
    'tanh': lambda x: np.tanh(x)
}

MAX_NODES_FOR_VISUALIZATION = 20
MAX_EDGES_FOR_VISUALIZATION = 30
COEFFICIENT_THRESHOLD = 1e-3  # Threshold for considering a term significant

class VisualizationError(Exception):
    """Raised when a formula is too complex to visualize clearly."""
    pass

def filter_significant_terms(coefficients, terms, threshold=COEFFICIENT_THRESHOLD):
    """Filter out terms with small coefficients."""
    significant_indices = np.where(np.abs(coefficients) > threshold)[0]
    return coefficients[significant_indices], [terms[i] for i in significant_indices]

def calculate_formula_complexity(formula):
    """Calculate complexity score of a symbolic formula."""
    # Count number of operations and functions
    operations = formula.count('+') + formula.count('*')
    functions = sum(func in formula for func in ADVANCED_LIB.keys())
    return operations + functions

def get_model_predictions(model, X, mode):
    """Obtain model predictions for regression or classification."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(model.device)
    
    with torch.no_grad():
        preds = model(X)
        if mode == 'regression':
            return preds.cpu().numpy().flatten(), None
        elif mode == 'classification':
            if model.output_dim == 1:
                # Binary classification: return logits
                out = preds.cpu().numpy()
                return out.flatten(), out
            else:
                # Multi-class: return pre-softmax logits for all classes
                out = preds.cpu().numpy()
                # Use the logits directly for symbolic approximation
                return out.reshape(-1, model.output_dim), out
    raise ValueError("Unknown mode")

def build_design_matrix(X, return_names=False):
    """Construct design matrix from advanced nonlinear bases."""
    X_np = np.array(X)
    n_samples, d = X_np.shape
    F_parts = [np.ones((n_samples, 1))]
    names = ['1'] if return_names else None
    for j in range(d):
        xj = X_np[:, j:j+1]
        for key, func in ADVANCED_LIB.items():
            F_parts.append(func(xj))
            if return_names:
                names.append(f"{key}(x{j+1})")
    return (np.hstack(F_parts), names) if return_names else np.hstack(F_parts)

def extract_symbolic_formula(model, X, mode='regression'):
    """
    Extract a simplified symbolic formula focusing on significant terms.
    Handles regression, binary classification, and multi-class classification.
    """
    y_target, logits = get_model_predictions(model, X, mode)
    F, func_names = build_design_matrix(X, return_names=True)
    
    if mode == 'classification' and model.output_dim > 1:
        # Multi-class: create formula for each class
        formulas = []
        for class_idx in range(model.output_dim):
            class_logits = logits[:, class_idx]
            beta, _, _, _ = np.linalg.lstsq(F, class_logits, rcond=None)
            
            # Filter significant terms for this class
            significant_mask = np.abs(beta) > COEFFICIENT_THRESHOLD
            beta_filtered = beta[significant_mask]
            names_filtered = [func_names[i] for i, is_sig in enumerate(significant_mask) if is_sig]
            
            if len(names_filtered) == 0:
                formulas.append("0")
                continue
                
            # Build class-specific formula
            f1_terms = [f"({coef:.3f}*{name})" for coef, name 
                       in zip(beta_filtered, names_filtered)]
            class_formula = " + ".join(f1_terms)
            
            # Add to list of class formulas
            formulas.append(f"Class {class_idx}: {class_formula}")
        
        return "\n".join(formulas)
    
    else:
        # Binary classification or regression
        beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
        
        # Filter significant terms
        significant_mask = np.abs(beta) > COEFFICIENT_THRESHOLD
        beta_filtered = beta[significant_mask]
        names_filtered = [func_names[i] for i, is_sig in enumerate(significant_mask) if is_sig]
        
        if len(names_filtered) == 0:
            return "0"
            
        # Build first layer formula
        layer1_terms = [f"({coef:.3f}*{name})" for coef, name 
                       in zip(beta_filtered, names_filtered)]
        layer1_formula = " + ".join(layer1_terms)
        
        # For binary classification, wrap in sigmoid
        if mode == 'classification':
            return f"sigmoid({layer1_formula})"
        
        return layer1_formula

@functools.lru_cache(maxsize=32)
def get_symbolic_function(formula_str):
    """
    Generate a Python function from a symbolic formula string.
    The function expects the pre-computed first layer value as input.
    """
    def symbolic_func(f1):
        return eval(formula_str, {"np": np, "tanh": np.tanh, "f1": f1})
    return symbolic_func

def symbolic_function(model, X, mode='regression'):
    """
    Create a symbolic function based on the two-layer formula.
    This function reconstructs the two-layer operation.
    """
    F, _ = build_design_matrix(X, return_names=True)
    y_target, _ = get_model_predictions(model, X, mode)
    beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
    f1 = F.dot(beta)
    G = np.hstack([np.ones((f1.shape[0], 1)), 
                   f1.reshape(-1, 1), 
                   np.tanh(f1).reshape(-1, 1)])
    alpha, _, _, _ = np.linalg.lstsq(G, y_target, rcond=None)
    # Build a lambda function string representing the two-layer transformation.
    formula_str = f"{alpha[0]:.4f} + {alpha[1]:.4f} * f1 + {alpha[2]:.4f} * np.tanh(f1)"
    return get_symbolic_function(formula_str)

def test_symbolic_formula(model, X, mode='regression'):
    """Evaluate the symbolic approximation against the model."""
    y_target, out = get_model_predictions(model, X, mode)
    F = build_design_matrix(X, return_names=False)
    
    if mode == 'classification' and model.output_dim > 1:
        # Handle multi-class case
        accuracies = []
        for class_idx in range(model.output_dim):
            # Get target logits for current class
            class_logits = out[:, class_idx]
            
            # Fit symbolic approximation for this class
            beta, _, _, _ = np.linalg.lstsq(F, class_logits, rcond=None)
            
            # Get symbolic predictions
            symbolic_logits = F.dot(beta)
            
            # Compare predictions
            symbolic_class = (symbolic_logits > 0).astype(int)
            actual_class = (class_logits > 0).astype(int)
            accuracy = np.mean(symbolic_class == actual_class)
            accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        print(f"\nSymbolic Formula vs OIKAN Multi-class Classification Similarity:")
        for i, acc in enumerate(accuracies):
            print(f"Class {i} Accuracy: {acc:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        return accuracies
        
    elif mode == 'classification':  # Binary classification
        # Fit symbolic approximation
        beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
        symbolic_vals = F.dot(beta)
        
        # Compare predictions
        symbolic_probs = 1 / (1 + np.exp(-symbolic_vals))
        model_probs = 1 / (1 + np.exp(-y_target))
        accuracy = np.mean(np.abs(symbolic_probs - model_probs) < 0.5)
        
        print(f"\nSymbolic Formula vs OIKAN Binary Classification Accuracy: {accuracy:.4f}")
        return accuracy
        
    else:  # Regression
        beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
        symbolic_vals = F.dot(beta)
        
        mse = np.mean((symbolic_vals - y_target) ** 2)
        mae = np.mean(np.abs(symbolic_vals - y_target))
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return mse, mae, rmse

def plot_symbolic_formula(model, X, mode='regression'):
    """Plot graph representation of formula with complexity checks."""
    formula = extract_symbolic_formula(model, X, mode)
    
    if mode == 'classification' and model.output_dim > 1:
        raise VisualizationError(
            "Visualization not supported for multi-class classification. "
            "The formula structure becomes too complex for meaningful visualization."
        )
    
    complexity = calculate_formula_complexity(formula)
    
    # Check visualization limits
    if complexity > MAX_NODES_FOR_VISUALIZATION:
        raise VisualizationError(
            f"Formula too complex to visualize clearly (complexity score: {complexity}). "
            f"Maximum recommended complexity is {MAX_NODES_FOR_VISUALIZATION}. "
            "Try using a simpler model or increasing the coefficient threshold."
        )
    
    X_np = np.array(X)
    n_samples, n_inputs = X_np.shape

    G = nx.DiGraph()
    # Define 4 layers: 0: Inputs, 1: First layer basis, 2: Second layer transforms, 3: Output.
    layers = {0: [], 1: [], 2: [], 3: []}

    # Parse the formula; expected pattern: (alpha0 + alpha1 * (term) + alpha2 * tanh(term))
    pattern = r"\(?([-\d\.]+)\)?\s*\+\s*([-\d\.]+)\s*\*\s*\((.*?)\)\s*\+\s*([-\d\.]+)\s*\*\s*tanh\(\s*(.*?)\s*\)"
    m = re.match(pattern, formula)
    if not m:
        print("Formula pattern not recognized.")
        return
    alpha0, alpha1, term1, alpha2, term2 = m.groups()
    first_layer_terms = term1.split(" + ") if term1 else []

    # Layer 0: Input nodes
    input_nodes = {f"x{i}" for i in range(1, n_inputs+1)}
    for inp in input_nodes:
        G.add_node(inp, layer=0)
        layers[0].append(inp)
    
    # Layer 1: First layer basis nodes from the extracted terms.
    for term in first_layer_terms:
        if "*" in term:
            func_node = term.split("*")[1].strip("() ")
            G.add_node(func_node, layer=1)
            layers[1].append(func_node)
    
    # Layer 2: Second layer transformation nodes (identity and tanh)
    node_id = "Identity"
    node_tanh = "tanh"
    G.add_node(node_id, layer=2)
    layers[2].append(node_id)
    G.add_node(node_tanh, layer=2)
    layers[2].append(node_tanh)
    
    # Layer 3: Output node
    output_node = "Output"
    G.add_node(output_node, layer=3)
    layers[3].append(output_node)
    
    # Add edges:
    for term in first_layer_terms:
        parts = term.strip("() ").split("*")
        if len(parts) == 2:
            coef = float(parts[0])
            func = parts[1].strip()
            inp = re.search(r"x\d+", func)
            inp = inp.group(0) if inp else "x1"
            G.add_edge(inp, func, weight=coef)
    for node in layers[1]:
        G.add_edge(node, node_id, weight=float(alpha1))
        G.add_edge(node, node_tanh, weight=float(alpha2))
    G.add_edge(node_id, output_node, weight=1.0)
    G.add_edge(node_tanh, output_node, weight=1.0)
    
    # Add check for edge count
    if G.number_of_edges() > MAX_EDGES_FOR_VISUALIZATION:
        raise VisualizationError(
            f"Too many connections to visualize clearly ({G.number_of_edges()} edges). "
            f"Maximum recommended edges is {MAX_EDGES_FOR_VISUALIZATION}. "
            "Try using a simpler model or increasing the coefficient threshold."
        )
    
    # Position nodes for 4 layers with better horizontal and vertical spacing
    pos = {}
    layer_x = {0: -1, 1: 2, 2: 5, 3: 8}
    for l, nodes in layers.items():
        n = len(nodes)
        # Ensure a minimum vertical spacing
        for i, node in enumerate(sorted(nodes)):
            y = 1 - (i + 1) / (n + 1)
            pos[node] = (layer_x[l], y)
    
    # Set different node sizes per layer
    sizes = {}
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer == 0:
            sizes[node] = 2500
        elif layer == 1:
            sizes[node] = 3000
        elif layer == 2:
            sizes[node] = 3500
        elif layer == 3:
            sizes[node] = 4000

    node_sizes = [sizes[node] for node in G.nodes()]
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer == 0:
            node_colors.append("red")
        elif layer == 1:
            node_colors.append("skyblue")
        elif layer == 2:
            node_colors.append("orange")
        elif layer == 3:
            node_colors.append("green")
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
            font_size=10, arrows=True, arrowstyle='->', arrowsize=30)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10)
    plt.title("OIKAN Symbolic Formula Graph (4-layered)")
    plt.axis("off")
    plt.show()

def extract_latex_formula(model, X, mode='regression'):
    """Return the symbolic formula formatted as LaTeX code."""
    formula = extract_symbolic_formula(model, X, mode)
    
    if mode == 'classification' and model.output_dim > 1:
        # Handle multi-class classification
        latex_formulas = []
        for line in formula.split('\n'):
            if not line.strip():
                continue
            class_idx, class_formula = line.split(': ', 1)
            # Convert the formula to LaTeX
            terms = class_formula.split(" + ")
            latex_terms = []
            for term in terms:
                term = term.strip("()")
                if '*' in term:
                    coeff, basis = term.split('*', 1)
                    coeff = float(coeff)
                    latex_terms.append(f"{abs(coeff):.3f} \\cdot {basis.strip()}")
                else:
                    latex_terms.append(term)
            
            latex_formula = " + ".join(latex_terms).replace("tanh", "\\tanh").replace("exp", "\\exp")
            latex_formulas.append(f"{class_idx}: {latex_formula}")
        
        # Use raw string for LaTeX cases environment
        return r"$$ \begin{cases} " + r" \\ ".join(latex_formulas) + r" \end{cases} $$"
    
    else:
        # Handle regression and binary classification
        terms = formula.split(" + ")
        latex_terms = []
        
        for term in terms:
            term = term.strip("()")
            if '*' in term:
                coeff, basis = term.split('*', 1)
                coeff = float(coeff)
                latex_terms.append(f"{abs(coeff):.3f} \\cdot {basis.strip()}")
            else:
                latex_terms.append(term)
        
        latex_formula = " + ".join(latex_terms).replace("tanh", "\\tanh").replace("exp", "\\exp")
        
        if mode == 'classification' and model.output_dim == 1:
            latex_formula = r"\sigma(" + latex_formula + r")"
            
        return f"$$ {latex_formula} $$"