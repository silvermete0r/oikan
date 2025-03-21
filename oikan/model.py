import torch
import torch.nn as nn
from .utils import BSplineBasis, FourierBasis
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
    def __init__(self, input_dim, hidden_units=10, basis_type='bsplines', 
                 bspline_num_knots=10, bspline_degree=3, fourier_num_frequencies=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.basis_type = basis_type

        if basis_type == 'bsplines':
            self.basis_functions = nn.ModuleList([
                BSplineBasis(num_knots=bspline_num_knots, degree=bspline_degree) for _ in range(input_dim)
            ])
            self.basis_output_dim = input_dim * (bspline_num_knots - bspline_degree - 1)
        elif basis_type == 'fourier':
            self.basis_functions = nn.ModuleList([
                FourierBasis(num_frequencies=fourier_num_frequencies) for _ in range(input_dim)
            ])
            self.basis_output_dim = input_dim * (2 * fourier_num_frequencies)
        elif basis_type == 'combo':
            self.basis_functions_bspline = nn.ModuleList([
                BSplineBasis(num_knots=bspline_num_knots, degree=bspline_degree) for _ in range(input_dim)
            ])
            self.basis_functions_fourier = nn.ModuleList([
                FourierBasis(num_frequencies=fourier_num_frequencies) for _ in range(input_dim)
            ])
            self.basis_output_dim = input_dim * ((bspline_num_knots - bspline_degree - 1) + (2 * fourier_num_frequencies))
        else:
            raise ValueError(f"Unsupported basis_type: {basis_type}")

        self.interaction_weights = nn.Parameter(torch.randn(input_dim, input_dim))
    
    def forward(self, x):
        if self.basis_type == 'combo':
            transformed_bspline = [bf(x[:, i].unsqueeze(1)) for i, bf in enumerate(self.basis_functions_bspline)]
            transformed_fourier = [bf(x[:, i].unsqueeze(1)) for i, bf in enumerate(self.basis_functions_fourier)]
            basis_output = torch.cat(transformed_bspline + transformed_fourier, dim=1)
        else:
            transformed = [bf(x[:, i].unsqueeze(1)) for i, bf in enumerate(self.basis_functions)]
            basis_output = torch.cat(transformed, dim=1)

        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.input_dim, 1)
        interaction_matrix = torch.sigmoid(self.interaction_weights)
        interaction_features = torch.bmm(x_reshaped.transpose(1, 2), 
                                         x_reshaped * interaction_matrix.unsqueeze(0))
        interaction_features = interaction_features.view(batch_size, -1)
        return torch.cat([basis_output, interaction_features], dim=1)
    
    def get_output_dim(self):
        return self.basis_output_dim + self.input_dim

class OIKAN(nn.Module):
    """Main model combining nonlinear transforms, projection, and interpretable layers."""
    def __init__(self, input_dim=None, output_dim=None, hidden_units=10, reduced_dim=32, 
                 basis_type='bsplines', bspline_num_knots=10, bspline_degree=3, 
                 fourier_num_frequencies=5, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_units = hidden_units
        self.basis_type = basis_type
        self.reduced_dim = reduced_dim
        self.bspline_num_knots = bspline_num_knots
        self.bspline_degree = bspline_degree
        self.fourier_num_frequencies = fourier_num_frequencies
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_initialized = False

    def initialize_from_data(self, X, y=None):
        """Initialize network architecture based on input data dimensions."""
        if self.is_initialized:
            return

        if not isinstance(X, torch.Tensor):
            raise DataTypeError("Input X must be a torch.Tensor")

        self.input_dim = self.input_dim or X.shape[1]
        
        if y is not None:
            if not isinstance(y, torch.Tensor):
                raise DataTypeError("Input y must be a torch.Tensor")
            if len(y.shape) == 1:
                self.output_dim = self.output_dim or 1
            else:
                self.output_dim = self.output_dim or y.shape[1]
        elif self.output_dim is None:
            raise InitializationError("output_dim must be specified if y is not provided")

        try:
            self.efficientkan = EfficientKAN(self.input_dim, self.hidden_units, self.basis_type,
                                           self.bspline_num_knots, self.bspline_degree, 
                                           self.fourier_num_frequencies).to(self.device)
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
            
            if self.output_dim == 1 and len(output.shape) > 2:
                output = output.squeeze(-1)
            return output
        except Exception as e:
            raise OikanError(f"Forward pass failed: {str(e)}")

    def fit(self, X, y, epochs=100, lr=0.01, verbose=True):
        """Train the model (scikit-learn style)."""
        from .trainer import train, train_classification
        
        # Convert inputs to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y) if len(y.shape) > 1 or y.dtype == float else torch.LongTensor(y)
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Determine if this is classification based on y
        is_classification = y.dtype == torch.long
        
        # Use appropriate training function
        if is_classification:
            train_classification(self, (X, y), epochs=epochs, lr=lr, verbose=verbose)
        else:
            train(self, (X, y), epochs=epochs, lr=lr, verbose=verbose)
        return self

    def predict(self, X):
        """Generate predictions (scikit-learn style)."""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            
            # For classification, return class predictions
            if self.output_dim > 1:  # Multi-class classification
                return torch.argmax(outputs, dim=1).cpu().numpy()
            elif outputs.shape[1] == 1:  # Regression or binary classification
                return outputs.cpu().numpy().squeeze()
            
        return outputs.cpu().numpy()

    def predict_proba(self, X):
        """Generate probability predictions (for classification)."""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            # Apply softmax for multi-class or sigmoid for binary
            if self.output_dim > 1:
                proba = torch.softmax(outputs, dim=1)
            else:
                proba = torch.sigmoid(outputs)
        return proba.cpu().numpy()

    def extract_symbolic_formula(self, X):
        """Extract a symbolic formula representation of the model's predictions."""
        from .symbolic import extract_symbolic_formula
        self.eval()
        mode = 'regression' if self.output_dim == 1 else 'classification'
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return extract_symbolic_formula(self, X, mode=mode)

    def test_symbolic_formula(self, X):
        """Test the accuracy of the symbolic formula approximation."""
        from .symbolic import test_symbolic_formula
        self.eval()
        mode = 'regression' if self.output_dim == 1 else 'classification'
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return test_symbolic_formula(self, X, mode=mode)

    def plot_symbolic_formula(self, X):
        """Visualize the symbolic formula as a graph."""
        from .symbolic import plot_symbolic_formula
        self.eval()
        mode = 'regression' if self.output_dim == 1 else 'classification'
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return plot_symbolic_formula(self, X, mode=mode)

    def extract_latex_formula(self, X):
        """Extract the symbolic formula in LaTeX format."""
        from .symbolic import extract_latex_formula
        self.eval()
        mode = 'regression' if self.output_dim == 1 else 'classification'
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return extract_latex_formula(self, X, mode=mode)