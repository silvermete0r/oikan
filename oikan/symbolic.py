import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functools
import re

ADVANCED_LIB = {
    'x':    lambda x: x,
    'x^2':  lambda x: x**2,
    'x^3':  lambda x: x**3,
    'x^4':  lambda x: x**4,
    'x^5':  lambda x: x**5,
    'exp':  lambda x: np.exp(x),
    'log':  lambda x: np.log(np.abs(x) + 1e-8),
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'tanh': lambda x: np.tanh(x),
    'sin':  lambda x: np.sin(x),
    'abs':  lambda x: np.abs(x)
}

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
    Extract a two-layered symbolic formula.
    First layer: compute a linear combination of advanced nonlinear basis functions.
    Second layer: apply identity and tanh transformations.
    Final representation: output = alpha0 + alpha1 * (first layer) + alpha2 * tanh(first layer)
    """
    y_target, _ = get_model_predictions(model, X, mode)
    F, func_names = build_design_matrix(X, return_names=True)
    beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
    layer1_terms = [f"({coef:.2f}*{name})" for coef, name in zip(beta, func_names) if abs(coef) > 1e-4]
    layer1_formula = " + ".join(layer1_terms) if layer1_terms else "0"
    f1 = F.dot(beta)
    # Second layer using identity and tanh functions
    G = np.hstack([np.ones((f1.shape[0], 1)), 
                   f1.reshape(-1, 1), 
                   np.tanh(f1).reshape(-1, 1)])
    alpha, _, _, _ = np.linalg.lstsq(G, y_target, rcond=None)
    final_formula = f"({alpha[0]:.2f} + {alpha[1]:.2f} * ({layer1_formula}) + {alpha[2]:.2f} * tanh({layer1_formula}))"
    return final_formula

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
    """Evaluate the symbolic approximation against the model using the full two-layer representation."""
    y_target, out = get_model_predictions(model, X, mode)
    F = build_design_matrix(X, return_names=False)
    beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
    f1 = F.dot(beta)
    G = np.hstack([np.ones((f1.shape[0], 1)),
                   f1.reshape(-1, 1),
                   np.tanh(f1).reshape(-1, 1)])
    alpha, _, _, _ = np.linalg.lstsq(G, y_target, rcond=None)
    symbolic_vals = G.dot(alpha)
    
    if mode == 'regression':
        mse = np.mean((symbolic_vals - y_target) ** 2)
        mae = np.mean(np.abs(symbolic_vals - y_target))
        rmse = np.sqrt(mse)
        print("\nSymbolic Formula vs OIKAN Regression Metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return mse, mae, rmse
    elif mode == 'classification':
        if model.output_dim == 1:
            symbolic_probs = 1 / (1 + np.exp(-symbolic_vals))
            model_probs = 1 / (1 + np.exp(-y_target))
            accuracy = np.mean(np.abs(symbolic_probs - model_probs) < 0.5)
        else:
            sym_preds = np.argmax(symbolic_vals.reshape(-1, model.output_dim), axis=1)
            model_preds = np.argmax(out, axis=1)
            accuracy = np.mean(sym_preds == model_preds)
        print(f"\nSymbolic Formula vs OIKAN Classification Similarity: {accuracy:.4f}")
        return accuracy

def plot_symbolic_formula(model, X, mode='regression'):
    """
    Plot a 4-layer graph:
      Layer 0: Inputs
      Layer 1: First layer basis nodes (advanced functions)
      Layer 2: Second layer transformation nodes (identity and tanh)
      Layer 3: Output
    Edge labels show assigned coefficients.
    """
    formula = extract_symbolic_formula(model, X, mode)
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
    terms = formula.split(" + ")
    latex_terms = []
    for term in terms:
        expr = term.strip("()")
        coeff_str, basis = expr.split("*", 1) if "*" in expr else (expr, "")
        coeff = float(coeff_str)
        missing = basis.count("(") - basis.count(")")
        if missing > 0:
            basis = basis + ")" * missing
        coeff_latex = f"{abs(coeff):.2f}".rstrip("0").rstrip(".")
        term_latex = coeff_latex if basis.strip() == "0" else f"{coeff_latex} \\cdot {basis.strip()}"
        latex_terms.append(f"- {term_latex}" if coeff < 0 else f"+ {term_latex}")
    latex_formula = " ".join(latex_terms).lstrip("+ ").strip()
    return f"$$ {latex_formula} $$"