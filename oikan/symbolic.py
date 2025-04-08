import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

ADVANCED_LIB = {
    'x':    lambda x: x,
    'x^2':  lambda x: x**2,
    'x^3':  lambda x: x**3,
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
    
    with torch.no_grad():
        preds = model(X)
    if mode == 'regression':
        return preds.detach().cpu().numpy().flatten(), None
    elif mode == 'classification':
        out = preds.detach().cpu().numpy()
        target = (out[:, 0] - out[:, 1]).flatten() if out.shape[1] > 1 else out.flatten()
        return target, out
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
    """Approximate a symbolic formula representing the model."""
    y_target, _ = get_model_predictions(model, X, mode)
    F, func_names = build_design_matrix(X, return_names=True)
    beta, _, _, _ = np.linalg.lstsq(F, y_target, rcond=None)
    terms = [f"({c:.2f}*{name})" for c, name in zip(beta, func_names) if abs(c) > 1e-4]
    return " + ".join(terms)

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
    """Plot a 3-layer graph: Inputs -> Function nodes -> Output, with edge labels showing coefficients."""
    import re
    formula = extract_symbolic_formula(model, X, mode)
    X_np = np.array(X)
    n_samples, n_inputs = X_np.shape

    G = nx.DiGraph()
    # Define 3 layers: 0: Inputs, 1: Function nodes, 2: Output.
    layers = {0: [], 1: [], 2: []}
    
    # Process symbolic formula terms into edges: (input, function_node, coefficient)
    function_edges = []
    terms = formula.split(" + ")
    for term in terms:
        term_clean = term.strip()
        if term_clean.startswith("(") and term_clean.endswith(")"):
            term_clean = term_clean[1:-1]
        if "*" in term_clean:
            coef_str, basis = term_clean.split("*", 1)
            try:
                coef = float(coef_str)
            except:
                continue
            if basis.strip() == "1":
                # Constant term: use dummy input "Const" and function node "1"
                function_edges.append(("Const", "1", coef))
            else:
                m = re.match(r'([^\(]+)\(x(\d+)\)', basis.strip())
                if m:
                    func_name, input_idx = m.groups()
                    function_edges.append((f"x{input_idx}", f"{func_name}(x{input_idx})", coef))
        else:
            try:
                coef = float(term_clean)
                function_edges.append(("Const", "1", coef))
            except:
                continue

    # Layer 0: Input nodes (red circles): include x1...xn and dummy "Const" if needed.
    input_nodes = {f"x{i}" for i in range(1, n_inputs+1)}
    for inp in input_nodes:
        G.add_node(inp, layer=0)
        layers[0].append(inp)
    if any(inp == "Const" for inp, _, _ in function_edges):
        G.add_node("Const", layer=0)
        layers[0].append("Const")
    
    # Layer 1: Function nodes (skyblue): unique function nodes from edges.
    function_set = {func for _, func, _ in function_edges}
    for func in function_set:
        G.add_node(func, layer=1)
        layers[1].append(func)
    
    # Layer 2: Output node (green circle)
    output_node = "Output"
    G.add_node(output_node, layer=2)
    layers[2].append(output_node)
    
    # Add edges: from input to function and from function to output (with coefficient as edge label)
    for inp, func, coef in function_edges:
        G.add_edge(inp, func)
        G.add_edge(func, output_node, weight=coef)
    
    # Adjust positions with increased horizontal and vertical spacing.
    pos = {}
    
    layer_x = {0: 0, 1: 5, 2: 10}
    for l, nodes in layers.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            y = 0.9 - (i * (0.8 / (n - 1))) if n > 1 else 0.5
            pos[node] = (layer_x[l], y)
    
    # Update node sizes.
    sizes = {}
    for node in G.nodes():
        layer = G.nodes[node].get('layer')
        if layer == 0:
            sizes[node] = 3000
        elif layer == 1:
            sizes[node] = 3500
        elif layer == 2:
            sizes[node] = 4000
    node_sizes = [sizes[node] for node in G.nodes()]
    
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node].get('layer', -1)
        if layer == 0:
            node_colors.append("red")
        elif layer == 2:
            node_colors.append("green")
        else:
            node_colors.append("skyblue")
    
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
            font_size=10, arrows=True, arrowstyle='->', arrowsize=25)
    
    # Label only edges from function nodes to output with the coefficient value.
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    plt.title("OIKAN Symbolic Formula Graph")
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
        term_latex = coeff_latex if basis.strip() == "1" else f"{coeff_latex} \\cdot {basis.strip()}"
        latex_terms.append(f"- {term_latex}" if coeff < 0 else f"+ {term_latex}")
    latex_formula = " ".join(latex_terms).lstrip("+ ").strip()
    return f"$$ {latex_formula} $$"