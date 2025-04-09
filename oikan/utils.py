from .exceptions import *
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import LassoCV
from sympy import symbols, expand

# Modified basis functions with explicit variable notation
ADVANCED_LIB = {
    'x':    ('x', lambda x: x),
    'x^2':  ('x^2', lambda x: np.clip(x**2, -100, 100)),
    'x^3':  ('x^3', lambda x: np.clip(x**3, -100, 100)),
    'exp':  ('exp(x)', lambda x: np.exp(np.clip(x, -10, 10))),
    'log':  ('log(x)', lambda x: np.log(np.abs(x) + 1)),
    'sqrt': ('sqrt(x)', lambda x: np.sqrt(np.abs(x))),
    'tanh': ('tanh(x)', lambda x: np.tanh(x)),
    'sin':  ('sin(x)', lambda x: np.sin(np.clip(x, -10*np.pi, 10*np.pi))),
    'abs':  ('abs(x)', lambda x: np.abs(x))
}

def normalize_data(x):
    """Normalize data to [-1, 1] range"""
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    x_range = (x_max - x_min)
    x_range[x_range == 0] = 1  # Prevent division by zero
    return 2 * (x - x_min) / x_range - 1

class EdgeActivation(nn.Module):
    """Learnable edge-based activation function."""
    def __init__(self, num_basis=10):
        super().__init__()
        self.num_basis = num_basis
        self.weights = nn.Parameter(torch.randn(num_basis))
        self.centers = torch.linspace(-1, 1, num_basis)
        self.width = 2.0 / (num_basis - 1)
        
    def forward(self, x):
        # Compute RBF basis functions
        x_expanded = x.unsqueeze(-1)
        centers = self.centers.to(x.device)
        rbf = torch.exp(-(x_expanded - centers)**2 / (2 * self.width**2))
        return torch.matmul(rbf, self.weights)
    
    def get_symbolic_repr(self, threshold=1e-4):
        """Get symbolic representation of the activation function."""
        significant_terms = []
        x = symbols('x')
        
        for i, weight in enumerate(self.weights.detach()):
            if abs(weight) > threshold:
                center = float(self.centers[i])
                term = weight * torch.exp(-(x - center)**2 / (2 * self.width**2))
                significant_terms.append(str(term))
                
        return " + ".join(significant_terms)

class BSplineBasis(nn.Module):
    """Enhanced B-Spline basis with advanced symbolic formula extraction."""
    def __init__(self, num_knots=10, degree=3):
        super().__init__()
        if num_knots < degree + 5:
            raise ValueError(f"Number of knots must be at least {degree + 5}")
        
        self.num_knots = num_knots
        self.degree = degree
        
        # Create knot vector
        inner_knots = np.linspace(0, 1, num_knots - 2 * degree)
        self.knots = np.concatenate([
            np.zeros(degree),
            inner_knots,
            np.ones(degree)
        ])
        self.register_buffer('knots_tensor', torch.FloatTensor(self.knots))
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        # Normalize input to [0,1]
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Compute B-spline basis values
        basis_values = []
        for i in range(len(self.knots) - self.degree - 1):
            spl = BSpline.basis_element(self.knots[i:i+self.degree+2])
            basis_i = torch.FloatTensor(spl(x_norm.detach().cpu().numpy().ravel()))
            basis_values.append(basis_i)
            
        return torch.stack(basis_values, dim=1).to(x.device)
    
    def get_symbolic_approximation(self, x, y, max_terms=5, threshold=1e-4):
        """Extract interpretable symbolic approximation using advanced basis functions."""
        # Generate features using advanced basis functions
        features = []
        feature_names = []
        
        # Add constant term
        features.append(np.ones_like(x))
        feature_names.append("1")
        
        # Add advanced basis function features
        for name, func in ADVANCED_LIB.values():
            feat = func(x.ravel())
            if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                features.append(feat)
                feature_names.append(name)
        
        # Stack features and fit Lasso
        X_features = np.column_stack(features)
        lasso = LassoCV(cv=5)
        lasso.fit(X_features, y.ravel())
        
        # Build symbolic formula from significant terms
        terms = []
        for coef, name in zip(lasso.coef_, feature_names):
            if abs(coef) > threshold:
                if name == "1":
                    terms.append(f"{coef:.4f}")
                else:
                    terms.append(f"{coef:.4f}*{name}")
                    
        return " + ".join(terms) if terms else "0"

class SymbolicEdge(nn.Module):
    """Edge-based activation function learner with advanced basis functions."""
    def __init__(self, input_dim=1, num_basis=10):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(len(ADVANCED_LIB)) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        x_normalized = normalize_data(x.detach().cpu().numpy())
        features = []
        for _, func in ADVANCED_LIB.values():
            feat = torch.tensor(func(x_normalized), 
                              dtype=torch.float32).to(x.device)
            features.append(feat)
        
        features = torch.stack(features, dim=-1)
        return torch.matmul(features, self.weights.unsqueeze(0).T) + self.bias
    
    def get_symbolic_repr(self, threshold=1e-4):
        """Get symbolic representation using advanced basis functions."""
        terms = []
        for w, (notation, _) in zip(self.weights, ADVANCED_LIB.items()):
            if abs(w.item()) > threshold:
                terms.append(f"{w.item():.4f}*{notation[0]}")
        
        if abs(self.bias.item()) > threshold:
            terms.append(f"{self.bias.item():.4f}")
            
        return " + ".join(terms) if terms else "0"