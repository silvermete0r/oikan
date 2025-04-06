import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator
from .utils import BSplineBasis
from .exceptions import *

class AdaptiveBasisLayer(nn.Module):
    """Applies a linear transformation in the interpretable model."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

class EfficientKAN(nn.Module):
    """Transforms features using nonlinear bases and captures interaction terms."""
    def __init__(self, input_dim, hidden_units=10, bspline_num_knots=10, bspline_degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        # Use only BSpline basis functions by default
        self.basis_functions = nn.ModuleList([
            BSplineBasis(num_knots=bspline_num_knots, degree=bspline_degree)
            for _ in range(input_dim)
        ])
        self.basis_output_dim = input_dim * (bspline_num_knots - bspline_degree - 1)
        self.interaction_weights = nn.Parameter(torch.randn(input_dim, input_dim))
    
    def forward(self, x):
        # Vectorized basis transformation for all features using torch.stack
        # x shape: (batch, input_dim)
        basis_outputs = torch.stack(
            [self.basis_functions[i](x[:, i:i+1]) for i in range(self.input_dim)],
            dim=1  # Result: (batch, input_dim, n_basis)
        )
        batch_size, _, n_basis = basis_outputs.shape
        basis_output = basis_outputs.view(batch_size, self.input_dim * n_basis)
        # Compute interaction features (unchanged)
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.input_dim, 1)
        interaction_matrix = torch.sigmoid(self.interaction_weights)
        interaction_features = torch.bmm(
            x_reshaped.transpose(1, 2),
            x_reshaped * interaction_matrix.unsqueeze(0)
        )
        interaction_features = interaction_features.view(batch_size, -1)
        return torch.cat([basis_output, interaction_features], dim=1)
    
    def get_output_dim(self):
        return self.basis_output_dim + self.input_dim

class OIKAN(BaseEstimator, nn.Module):
    """Main model combining nonlinear transforms, projection, and interpretable layers."""
    def __init__(self, mode='regression', reduced_dim=32, hidden_units=None, 
                 bspline_degree=3, bspline_num_knots=None):
        super().__init__()
        # Automatically choose device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.reduced_dim = reduced_dim
        self.hidden_units = hidden_units
        self.bspline_degree = bspline_degree
        self.bspline_num_knots = bspline_num_knots
        # Auto-tuned dimensions
        self.input_dim = None  
        self.output_dim = None
        self.is_initialized = False

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'mode': self.mode,
            'reduced_dim': self.reduced_dim,
            'hidden_units': self.hidden_units,
            'bspline_degree': self.bspline_degree,
            'bspline_num_knots': self.bspline_num_knots
        }

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def initialize_from_data(self, X, y=None):
        """Initialize network architecture based on input data dimensions."""
        if self.is_initialized:
            return

        if not isinstance(X, torch.Tensor):
            raise DataTypeError("Input X must be a torch.Tensor")
        self.input_dim = self.input_dim or X.shape[1]
        
        # Auto-tune hyperparameters based on input data
        if self.hidden_units is None:
            self.hidden_units = max(10, self.input_dim * 2)
        if self.bspline_degree is None:
            self.bspline_degree = 3
        if self.bspline_num_knots is None:
            self.bspline_num_knots = max(10, self.bspline_degree + 5)
        
        # Determine output_dim based on y if available
        if y is not None:
            if not isinstance(y, torch.Tensor):
                raise DataTypeError("Input y must be a torch.Tensor")
            if self.mode == 'classification':
                n_classes = len(torch.unique(y))
                self.output_dim = 1 if n_classes == 2 else n_classes
            else:
                self.output_dim = self.output_dim or (1 if len(y.shape) == 1 else y.shape[1])
        elif self.output_dim is None:
            raise InitializationError("output_dim must be specified if y is not provided")

        try:
            # Instantiate EfficientKAN with auto-tuned hyperparameters
            self.efficientkan = EfficientKAN(self.input_dim, self.hidden_units, 
                                             self.bspline_num_knots, self.bspline_degree).to(self.device)
            feature_dim = self.efficientkan.get_output_dim()
            self.svd_projection = nn.Linear(feature_dim, self.reduced_dim, bias=False).to(self.device)
            self.interpretable_layers = nn.Sequential(
                AdaptiveBasisLayer(self.reduced_dim, 32),
                nn.ReLU(),
                AdaptiveBasisLayer(32, self.output_dim)
            ).to(self.device)
            self.is_initialized = True
        except Exception as e:
            raise InitializationError(f"Failed to initialize model components: {str(e)}")

    def forward(self, x):
        if not self.is_initialized:
            try:
                self.initialize_from_data(x)
            except Exception as e:
                raise InitializationError(f"Failed to initialize model: {str(e)}")

        if not isinstance(x, torch.Tensor):
            raise DataTypeError("Input must be a torch.Tensor")

        if x.shape[1] != self.input_dim:
            raise DimensionalityError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")

        try:
            transformed_x = self.efficientkan(x)
            transformed_x = self.svd_projection(transformed_x)
            output = self.interpretable_layers(transformed_x)
            
            return output
            
        except Exception as e:
            raise OikanError(f"Forward pass failed: {str(e)}")

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True, history=False):
        """Train the model following scikit-learn's API."""
        # Convert numpy arrays to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X.astype('float32'))
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long if self.mode == 'classification' else torch.float32)
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Initialize if not already done
        if not self.is_initialized:
            self.initialize_from_data(X, y)
        
        # Use appropriate training function with history tracking
        if self.mode == 'classification':
            from .trainer import train_classification
            model, training_history = train_classification(self, (X, y), epochs=epochs, lr=lr, verbose=verbose)
        else:
            from .trainer import train
            model, training_history = train(self, (X, y), epochs=epochs, lr=lr, verbose=verbose)
        
        return (self, training_history) if history else self

    def predict(self, X):
        """Generate predictions following scikit-learn's API."""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X.astype('float32'))
        X = X.to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            
            if self.mode == 'classification':
                if self.output_dim == 1:
                    return (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).reshape(-1)
                else:
                    return torch.argmax(outputs, dim=1).cpu().numpy()
            
            return outputs.cpu().numpy()

    def predict_proba(self, X):
        """Probability estimates for classification following scikit-learn's API."""
        if self.mode != 'classification':
            raise ValueError("predict_proba is only available for classification mode")
            
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X.astype('float32'))
        X = X.to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if self.output_dim == 1:
                proba = torch.sigmoid(outputs)
                return np.column_stack([1 - proba.cpu().numpy(), proba.cpu().numpy()])
            else:
                return torch.softmax(outputs, dim=1).cpu().numpy()

    def extract_symbolic_formula(self, X):
        """Extract a symbolic formula representation of the model's predictions."""
        from .symbolic import extract_symbolic_formula
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return extract_symbolic_formula(self, X, mode=self.mode)

    def test_symbolic_formula(self, X):
        """Test the accuracy of the symbolic formula approximation."""
        from .symbolic import test_symbolic_formula
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return test_symbolic_formula(self, X, mode=self.mode)

    def plot_symbolic_formula(self, X):
        """Visualize the symbolic formula as a graph."""
        from .symbolic import plot_symbolic_formula
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return plot_symbolic_formula(self, X, mode=self.mode)

    def extract_latex_formula(self, X):
        """Extract the symbolic formula in LaTeX format."""
        from .symbolic import extract_latex_formula
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return extract_latex_formula(self, X, mode=self.mode)