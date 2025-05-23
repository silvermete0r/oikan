import json
import numpy as np

def softmax(X):
    """
    Computes the softmax of the input array.
    
    Parameters:
    -----------
    X : array-like
        Input data.
    
    Returns:
    --------
    softmax_X : ndarray
        Softmax of the input data.
    """
    X -= np.max(X, axis=1, keepdims=True)
    e_X = np.exp(X)
    return e_X / np.sum(e_X, axis=1, keepdims=True)

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

def predict(X, symbolic_model):
    """
    Predicts class labels for the input data.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    
    Returns:
    --------
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.
    """
    X = np.asarray(X)
    X_transformed = evaluate_basis_functions(X, symbolic_model['basis_functions'], 
                                            symbolic_model['n_features'])
    logits = np.dot(X_transformed, np.array(symbolic_model['coefficients_list']).T)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)

if __name__ == "__main__":
    with open('outputs/iris_model.json', 'r') as f:
        symbolic_model = json.load(f)
    X = np.array([[5.1, 3.5, 1.4, 0.2],
                  [7.0, 3.2, 4.7, 1.4],
                  [6.3, 3.3, 6.0, 2.5]])
    y_pred = predict(X, symbolic_model)
    print(y_pred)