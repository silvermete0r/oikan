import numpy as np

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