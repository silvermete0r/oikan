import numpy as np
import sympy as sp
import json

def evaluate_basis_functions(X, basis_functions, n_features):
    """
    Evaluates basis functions on the input data.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    basis_functions : list
        List of basis function strings (e.g., '1', 'x0', 'x0^2', 'x0 x1', 'log1p_x0').
    n_features : int
        Number of input features.
    
    Returns:
    --------
    X_transformed : ndarray of shape (n_samples, n_basis_functions)
        Transformed data matrix.
    """
    X_transformed = np.zeros((X.shape[0], len(basis_functions)))
    for i, func in enumerate(basis_functions):
        if func == '1':
            X_transformed[:, i] = 1
        elif func.startswith('log1p_x'):
            idx = int(func.split('_')[1][1:])
            X_transformed[:, i] = np.log1p(np.abs(X[:, idx]))
        elif func.startswith('exp_x'):
            idx = int(func.split('_')[1][1:])
            X_transformed[:, i] = np.exp(np.clip(X[:, idx], -10, 10))
        elif func.startswith('sin_x'):
            idx = int(func.split('_')[1][1:])
            X_transformed[:, i] = np.sin(X[:, idx])
        elif '^' in func:
            var, power = func.split('^')
            idx = int(var[1:])
            X_transformed[:, i] = X[:, idx] ** int(power)
        elif ' ' in func:
            vars = func.split(' ')
            result = np.ones(X.shape[0])
            for var in vars:
                idx = int(var[1:])
                result *= X[:, idx]
            X_transformed[:, i] = result
        else:
            idx = int(func[1:])
            X_transformed[:, i] = X[:, idx]
    return X_transformed

def get_features_involved(basis_function):
    """
    Extracts the feature indices involved in a basis function string.
    
    Parameters:
    -----------
    basis_function : str
        String representation of the basis function, e.g., 'x0', 'x0^2', 'x0 x1', 'log1p_x0'.
    
    Returns:
    --------
    set : Set of feature indices involved.
    """
    if basis_function == '1':
        return set()
    features = set()
    if '_' in basis_function:  # Handle non-linear functions like 'log1p_x0'
        parts = basis_function.split('_')
        if len(parts) == 2 and parts[1].startswith('x'):
            idx = int(parts[1][1:])
            features.add(idx)
    elif '^' in basis_function:  # Handle powers, e.g., 'x0^2'
        var = basis_function.split('^')[0]
        idx = int(var[1:])
        features.add(idx)
    elif ' ' in basis_function:  # Handle interactions, e.g., 'x0 x1'
        for part in basis_function.split():
            idx = int(part[1:])
            features.add(idx)
    elif basis_function.startswith('x'):
        idx = int(basis_function[1:])
        features.add(idx)
    return features

def sympify_formula(basis_functions, coefficients, n_features, threshold=0.00005):
    """
    Sympifies a symbolic formula using SymPy.
    
    Parameters:
    -----------
    basis_functions : list
        List of basis function strings (e.g., 'x0', 'x0^2', 'x0 x1', 'exp_x0').
    coefficients : list
        List of coefficients corresponding to each basis function.
    n_features : int
        Number of input features.
    threshold : float, optional (default=0.00005)
        Coefficients with absolute value below this are excluded.
    
    Returns:
    --------
    str
        Sympified formula as a string, or '0' if empty.
    """
    # Define symbolic variables
    x = sp.symbols(f'x0:{n_features}')
    expr = 0
    
    # Build the expression
    for coef, func in zip(coefficients, basis_functions):
        if abs(coef) < threshold:
            continue  # Skip negligible coefficients
        if func == '1':
            term = coef
        elif func.startswith('log1p_x'):
            idx = int(func.split('_')[1][1:])
            term = coef * sp.log(1 + sp.Abs(x[idx]))
        elif func.startswith('exp_x'):
            idx = int(func.split('_')[1][1:])
            term = coef * sp.exp(x[idx])
        elif func.startswith('sin_x'):
            idx = int(func.split('_')[1][1:])
            term = coef * sp.sin(x[idx])
        elif '^' in func:
            var, power = func.split('^')
            idx = int(var[1:])
            term = coef * x[idx]**int(power)
        elif ' ' in func:
            vars = func.split(' ')
            term = coef
            for var in vars:
                idx = int(var[1:])
                term *= x[idx]
        else:
            idx = int(func[1:])
            term = coef * x[idx]
        expr += term
    
    # Sympify the expression
    sympified_expr = sp.simplify(expr)
    
    # Convert to string with rounded coefficients
    def format_term(term):
        if term.is_Mul:
            coeff = 1
            factors = []
            for factor in term.args:
                if factor.is_Number:
                    coeff *= float(factor)
                else:
                    factors.append(str(factor))
            if abs(coeff) < threshold:
                return None
            return f"{coeff:.5f}*{'*'.join(factors)}" if factors else f"{coeff:.5f}"
        elif term.is_Add:
            return None  # Handle in recursion
        elif term.is_Number:
            return f"{float(term):.5f}" if abs(float(term)) >= threshold else None
        else:
            return f"{1.0:.5f}*{term}" if abs(1.0) >= threshold else None

    terms = []
    if sympified_expr.is_Add:
        for term in sympified_expr.args:
            formatted = format_term(term)
            if formatted:
                terms.append(formatted)
    else:
        formatted = format_term(sympified_expr)
        if formatted:
            terms.append(formatted)
    
    formula = " + ".join(terms).replace("+ -", "- ")
    return formula if formula else "0"

def get_latex_formula(basis_functions, coefficients, n_features, threshold=0.00005):
    """
    Generates a LaTeX formula from the basis functions and coefficients.
    
    Parameters:
    -----------
    basis_functions : list
        List of basis function strings (e.g., 'x0', 'x0^2', 'x0 x1', 'exp_x0').
    coefficients : list
        List of coefficients corresponding to each basis function.
    n_features : int
        Number of input features.
    threshold : float, optional (default=0.00005)
        Coefficients with absolute value below this are excluded.
    
    Returns:
    --------
    str
        LaTeX formula as a string, or '0' if empty.
    """
    formula = sympify_formula(basis_functions, coefficients, n_features, threshold)
    return sp.latex(sp.sympify(formula))

if __name__ == "__main__":
    with open('outputs/california_housing_model.json', 'r') as f:
        model = json.load(f)
    print('Sympified formula:', sympify_formula(model['basis_functions'], model['coefficients'], model['n_features']))
    print('LaTeX formula:', get_latex_formula(model['basis_functions'], model['coefficients'], model['n_features']))