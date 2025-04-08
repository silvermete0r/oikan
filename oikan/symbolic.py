import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

ADVANCED_LIB = {
    # Basic functions
    'x':    lambda x: x,
    'x^2':  lambda x: x**2,
    'exp':  lambda x: np.exp(x),
    'sin':  lambda x: np.sin(x),
    'tanh': lambda x: np.tanh(x),
    # Advanced functions
    'log':  lambda x: np.log(np.abs(x) + 1e-7),
    'sqrt': lambda x: np.sqrt(np.abs(x)),
    'relu': lambda x: np.maximum(0, x),
    'gaussian': lambda x: np.exp(-x**2),
    'polynomial3': lambda x: x**3
}

def get_model_predictions(model, X, mode):
    """Obtain model predictions with improved error handling."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    try:
        with torch.no_grad():
            preds = model(X)
        if mode == 'regression':
            return preds.detach().cpu().numpy().flatten(), None
        elif mode == 'classification':
            out = preds.detach().cpu().numpy()
            target = (out[:, 0] - out[:, 1]).flatten() if out.shape[1] > 1 else out.flatten()
            return target, out
        raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        raise RuntimeError(f"Failed to get predictions: {str(e)}")

def build_design_matrix(X, return_names=False, coefficient_threshold=1e-5):
    """Construct design matrix with improved numerical stability."""
    X_np = np.array(X)
    n_samples, d = X_np.shape
    F_parts = [np.ones((n_samples, 1))]
    names = ['1'] if return_names else None
    
    try:
        for j in range(d):
            xj = X_np[:, j:j+1]
            for key, func in ADVANCED_LIB.items():
                # Apply function with numerical stability checks
                try:
                    result = func(xj)
                    if np.all(np.isfinite(result)):
                        F_parts.append(result)
                        if return_names:
                            names.append(f"{key}(x{j+1})")
                except Exception:
                    continue  # Skip failed function applications
        
        F = np.hstack(F_parts)
        # Remove near-zero columns for better stability
        if not return_names:
            return F
        
        valid_cols = np.where(np.abs(F).mean(axis=0) > coefficient_threshold)[0]
        F = F[:, valid_cols]
        names = [names[i] for i in valid_cols]
        return F, names
    
    except Exception as e:
        raise RuntimeError(f"Failed to build design matrix: {str(e)}")

def extract_symbolic_formula(model, X, mode='regression', coef_threshold=1e-4):
    """Extract symbolic formula with improved coefficient selection."""
    y_target, _ = get_model_predictions(model, X, mode)
    F, func_names = build_design_matrix(X, return_names=True)
    
    try:
        # Use robust least squares with regularization
        beta, residuals, rank, s = np.linalg.lstsq(F, y_target, rcond=None)
        
        # Smart coefficient thresholding
        max_coef = np.max(np.abs(beta))
        rel_threshold = coef_threshold * max_coef
        significant_terms = [(c, name) for c, name in zip(beta, func_names) 
                           if abs(c) > rel_threshold]
        
        if not significant_terms:
            return "0"  # Return zero if no significant terms
        
        terms = []
        for coef, name in significant_terms:
            coef_str = f"{coef:.3f}".rstrip('0').rstrip('.')
            terms.append(f"({coef_str}*{name})")
        
        return " + ".join(terms)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract formula: {str(e)}")

def test_symbolic_formula(model, X, mode='regression'):
    """Evaluate the symbolic approximation against the model."""
    y_target, out = get_model_predictions(model, X, mode)
    F = build_design_matrix(X, return_names=False)
    beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
    symbolic_vals = F.dot(beta)
    if mode == 'regression':
        mse = np.mean((symbolic_vals - y_target) ** 2)
        mae = np.mean(np.abs(symbolic_vals - y_target))
        rmse = np.sqrt(mse)
        header_title = "Symbolic Formula & OIKAN Similarity"
        header = ["MSE", "MAE", "RMSE"]
        values = [mse, mae, rmse]
        col_width = 12
        row_sep = "+" + "+".join(["-" * (col_width + 2) for _ in header]) + "+"
        print(f"\n{header_title}")
        print("-" * len(header_title))
        print(row_sep)
        print("|" + "|".join(f" {h:^{col_width}} " for h in header) + "|")
        print(row_sep)
        print("|" + "|".join(f" {v:>{col_width}.4f} " for v in values) + "|")
        print(row_sep)
        return mse, mae, rmse
    elif mode == 'classification':
        sym_preds = np.where(symbolic_vals >= 0, 0, 1)
        model_classes = np.argmax(out, axis=1) if (out.ndim > 1) else (out >= 0.5).astype(int)
        if model_classes.shape[0] != sym_preds.shape[0]:
            raise ValueError("Shape mismatch between symbolic and model predictions.")
        accuracy = np.mean(sym_preds == model_classes)
        print(f"\nSymbolic Formula & OIKAN Similarity (Accuracy): {accuracy:.4f}")
        return accuracy

def plot_symbolic_formula(model, X, mode='regression'):
    """Plot formula graph with improved layout and readability."""
    try:
        formula = extract_symbolic_formula(model, X, mode)
        G = nx.DiGraph()
        
        # Create node layers
        input_nodes = set()
        func_nodes = set()
        coef_edges = []
        
        # Parse formula terms
        for term in formula.split(" + "):
            term = term.strip("()")
            if "*" in term:
                coef, expr = term.split("*", 1)
                coef = float(coef)
                
                if "(" in expr:
                    func, var = expr.split("(")
                    var = var.rstrip(")")
                    input_nodes.add(var)
                    func_nodes.add(f"{func}({var})")
                    G.add_edge(var, f"{func}({var})")
                    coef_edges.append((f"{func}({var})", coef))
                else:
                    func_nodes.add(expr)
                    coef_edges.append((expr, coef))
        
        # Add nodes with improved visual attributes
        for node in input_nodes:
            G.add_node(node, layer=0, color='lightblue')
        for node in func_nodes:
            G.add_node(node, layer=1, color='lightgreen')
        G.add_node("output", layer=2, color='salmon')
        
        # Add edges with coefficients
        for node, coef in coef_edges:
            G.add_edge(node, "output", weight=coef)
        
        # Improved layout
        pos = nx.multipartite_layout(G, subset_key="layer", align='vertical')
        
        # Enhanced visualization
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True,
                node_color=[G.nodes[node].get('color', 'white') for node in G.nodes()],
                node_size=2000, font_size=8, font_weight='bold',
                edge_color='gray', width=1, arrowsize=20)
        
        # Add edge labels with improved formatting
        edge_labels = {(u, v): f"{d['weight']:.3f}" 
                      for u, v, d in G.edges(data=True) if 'weight' in d}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                   font_size=8, font_color='red')
        
        plt.title("OIKAN Symbolic Formula Graph", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        raise RuntimeError(f"Failed to plot formula graph: {str(e)}")

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
        term_latex = coeff_latex if basis.strip() == "1" else f"{coeff_latex} \\cdot {basis.strip()}"
        latex_terms.append(f"- {term_latex}" if coeff < 0 else f"+ {term_latex}")
    latex_formula = " ".join(latex_terms).lstrip("+ ").strip()
    return f"$$ {latex_formula} $$"