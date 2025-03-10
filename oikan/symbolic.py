import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        preds = model(X_tensor)
    if mode == 'regression':
        return preds.detach().cpu().numpy().flatten(), None
    elif mode == 'classification':
        out = preds.detach().cpu().numpy()
        target = (out[:, 0] - out[:, 1]).flatten() if (out.ndim > 1 and out.shape[1] > 1) else out.flatten()
        return target, out
    else:
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
        print(f"(Advanced) MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return mse, mae, rmse
    elif mode == 'classification':
        sym_preds = np.where(symbolic_vals >= 0, 0, 1)
        model_classes = np.argmax(out, axis=1) if (out.ndim > 1) else (out >= 0.5).astype(int)
        if model_classes.shape[0] != sym_preds.shape[0]:
            raise ValueError("Shape mismatch between symbolic and model predictions.")
        accuracy = np.mean(sym_preds == model_classes)
        print(f"(Advanced) Accuracy: {accuracy:.4f}")
        return accuracy

def plot_symbolic_formula(model, X, mode='regression'):
    """Plot a graph representation of the extracted symbolic formula."""
    formula = extract_symbolic_formula(model, X, mode)
    G = nx.DiGraph()
    G.add_node("Output")
    terms = formula.split(" + ")
    for term in terms:
        expr = term.strip("()")
        coeff_str, basis = expr.split("*", 1) if "*" in expr else (expr, "unknown")
        node_label = f"{basis}\n({float(coeff_str):.2f})"
        G.add_node(node_label)
        G.add_edge(node_label, "Output", weight=float(coeff_str))
    left_nodes = [n for n in G.nodes() if n != "Output"]
    pos = {}
    n_left = len(left_nodes)
    for i, node in enumerate(sorted(left_nodes)):
        pos[node] = (0, 1 - (i / max(n_left - 1, 1)))
    pos["Output"] = (1, 0.5)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2500, font_size=10,
            arrows=True, arrowstyle='->', arrowsize=20)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
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