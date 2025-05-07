import numpy as np

def evaluate_basis_functions(X, basis_functions, n_features):
    """
    Evaluates basis functions on the input data.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    basis_functions : list
        List of basis function strings (e.g., '1', 'x0', 'x0^2', 'x0 x1').
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
        elif '^' in func:
            var, power = func.split('^')
            idx = int(var[1:])
            X_transformed[:, i] = X[:, idx] ** int(power)
        elif ' ' in func:
            var1, var2 = func.split(' ')
            idx1 = int(var1[1:])
            idx2 = int(var2[1:])
            X_transformed[:, i] = X[:, idx1] * X[:, idx2]
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
        String representation of the basis function, e.g., 'x0', 'x0^2', 'x0 x1'.
    
    Returns:
    --------
    set : Set of feature indices involved.
    """
    if basis_function == '1':  # Constant term involves no features
        return set()
    features = set()
    for part in basis_function.split():  # Split by space for interaction terms
        if part.startswith('x'):
            if '^' in part:  # Handle powers, e.g., 'x0^2'
                var = part.split('^')[0]  # Take 'x0'
            else:
                var = part  # Take 'x0' as is
            idx = int(var[1:])  # Extract index, e.g., 0
            features.add(idx)
    return features