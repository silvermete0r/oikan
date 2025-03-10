import torch
import torch.nn as nn
from .utils import BSplineBasis, FourierBasis

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
    def __init__(self, input_dim, output_dim, hidden_units=10, reduced_dim=32, 
                 basis_type='bsplines', forecast_mode=False,
                 bspline_num_knots=10, bspline_degree=3, fourier_num_frequencies=5):
        super().__init__()
        self.forecast_mode = forecast_mode
        if self.forecast_mode:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, batch_first=True)
            self.efficientkan = EfficientKAN(input_dim, hidden_units, basis_type,
                                              bspline_num_knots, bspline_degree, fourier_num_frequencies)
        else:
            self.efficientkan = EfficientKAN(input_dim, hidden_units, basis_type,
                                              bspline_num_knots, bspline_degree, fourier_num_frequencies)
        feature_dim = self.efficientkan.get_output_dim()
        self.svd_projection = nn.Linear(feature_dim, reduced_dim, bias=False)
        feature_dim = reduced_dim
        self.interpretable_layers = nn.Sequential(
            AdaptiveBasisLayer(feature_dim, 32),
            nn.ReLU(),
            AdaptiveBasisLayer(32, output_dim)
        )

    def forward(self, x):
        if self.forecast_mode:
            lstm_out, (hidden, _) = self.lstm(x)
            x_in = hidden[-1]
        else:
            x_in = x
        transformed_x = self.efficientkan(x_in)
        transformed_x = self.svd_projection(transformed_x)
        return self.interpretable_layers(transformed_x)