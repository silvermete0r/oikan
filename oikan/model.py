import torch
import torch.nn as nn
from .utils import BSplineBasis, FourierBasis

class AdaptiveBasisLayer(nn.Module):
    '''Layer that applies a linear transformation as part of interpretable modeling.'''
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x):
        # Linear transformation for adaptive basis processing
        return torch.matmul(x, self.weights) + self.bias

class EfficientKAN(nn.Module):
    '''Module computing feature transformations using nonlinear basis functions and interaction terms.'''
    def __init__(self, input_dim, hidden_units=10, basis_type='bspline'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.basis_type = basis_type
        
        if basis_type == 'bspline':
            # One BSpline per feature expected with adjusted output dimensions
            self.basis_functions = nn.ModuleList([BSplineBasis(hidden_units) for _ in range(input_dim)])
            self.basis_output_dim = input_dim * (hidden_units - 4)  # Adjusted for BSpline output
        elif basis_type == 'fourier':
            # Use Fourier basis transformation for each feature
            self.basis_functions = nn.ModuleList([FourierBasis(hidden_units//2) for _ in range(input_dim)])
            self.basis_output_dim = input_dim * hidden_units
            
        # Interaction layer: captures pairwise feature interactions
        self.interaction_weights = nn.Parameter(torch.randn(input_dim, input_dim))
    
    def forward(self, x):
        # Transform each feature using its corresponding basis function
        transformed_features = [bf(x[:, i].unsqueeze(1)) for i, bf in enumerate(self.basis_functions)]
        basis_output = torch.cat(transformed_features, dim=1)
        
        # Compute interaction features via fixed matrix multiplication
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.input_dim, 1)  # Reshape to [batch_size, input_dim, 1]
        interaction_matrix = torch.sigmoid(self.interaction_weights)  # Normalize interaction weights
        interaction_features = torch.bmm(x_reshaped.transpose(1, 2), 
                                       x_reshaped * interaction_matrix.unsqueeze(0))  # Result: [batch_size, 1, 1]
        interaction_features = interaction_features.view(batch_size, -1)  # Flatten interaction output
        
        return torch.cat([basis_output, interaction_features], dim=1)
    
    def get_output_dim(self):
        # Output dimension includes both basis and interaction features
        return self.basis_output_dim + self.input_dim

class OIKAN(nn.Module):
    '''Main OIKAN model combining nonlinear transformations, SVD-projection, and interpretable layers.'''
    def __init__(self, input_dim, output_dim, hidden_units=10, reduced_dim=32):
        super().__init__()
        self.efficientkan = EfficientKAN(input_dim, hidden_units)
        feature_dim = self.efficientkan.get_output_dim()
        
        # Apply SVD projection to compress high-dimensional features
        self.svd_projection = nn.Linear(feature_dim, reduced_dim, bias=False)
        feature_dim = reduced_dim  # Update feature dimension after projection
        
        # Interpretable layers for final mapping
        self.interpretable_layers = nn.Sequential(
            AdaptiveBasisLayer(feature_dim, 32),
            nn.ReLU(),
            AdaptiveBasisLayer(32, output_dim)
        )
    
    def forward(self, x):
        transformed_x = self.efficientkan(x)
        # Compress features prior to final prediction layer
        transformed_x = self.svd_projection(transformed_x)
        return self.interpretable_layers(transformed_x)