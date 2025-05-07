import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

def symbolic_regression(X, y, degree=2, alpha=0.1):
    """
    Performs symbolic regression on the input data.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    degree : int, optional (default=2)
        Maximum polynomial degree.
    alpha : float, optional (default=0.1)
        L1 regularization strength.
    
    Returns:
    --------
    dict : Contains 'basis_functions', 'coefficients' (or 'coefficients_list'), 'n_features', 'degree'
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    model = Lasso(alpha=alpha, fit_intercept=False)
    model.fit(X_poly, y)
    if len(y.shape) == 1 or y.shape[1] == 1:
        coef = model.coef_.flatten()
        selected_indices = np.where(np.abs(coef) > 1e-6)[0]
        return {
            'n_features': X.shape[1],
            'degree': degree,
            'basis_functions': poly.get_feature_names_out()[selected_indices].tolist(),
            'coefficients': coef[selected_indices].tolist()
        }
    else:
        coefficients_list = []
        selected_indices = set()
        for c in range(y.shape[1]):
            coef = model.coef_[c]
            indices = np.where(np.abs(coef) > 1e-6)[0]
            selected_indices.update(indices)
        selected_indices = list(selected_indices)
        basis_functions = poly.get_feature_names_out()[selected_indices].tolist()
        for c in range(y.shape[1]):
            coef = model.coef_[c]
            coef_selected = coef[selected_indices].tolist()
            coefficients_list.append(coef_selected)
        return {
            'n_features': X.shape[1],
            'degree': degree,
            'basis_functions': basis_functions,
            'coefficients_list': coefficients_list
        }