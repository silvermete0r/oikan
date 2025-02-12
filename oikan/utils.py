import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline

class BSplineBasis(nn.Module):
    def __init__(self, num_knots=10, degree=3):
        super().__init__()
        self.num_knots = max(num_knots, degree + 5)  # Ensure minimum number of knots
        self.degree = degree
        
        # Create knot vector with proper padding
        inner_knots = np.linspace(0, 1, self.num_knots - 2 * degree)
        left_pad = np.zeros(degree)
        right_pad = np.ones(degree)
        knots = np.concatenate([left_pad, inner_knots, right_pad])
        
        self.register_buffer('knots', torch.FloatTensor(knots))
        
    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        basis_values = np.zeros((x_np.shape[0], self.num_knots - self.degree - 1))
        
        # Normalize input to [0,1] range
        x_normalized = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        
        for i in range(self.num_knots - self.degree - 1):
            spl = BSpline.basis_element(self.knots[i:i+self.degree+2])
            basis_values[:, i] = spl(x_normalized.squeeze())
        
        # Replace NaN values with 0
        basis_values = np.nan_to_num(basis_values, 0)
        return torch.FloatTensor(basis_values).to(x.device)

class FourierBasis(nn.Module):
    def __init__(self, num_frequencies=5):
        super().__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, x):
        frequencies = torch.arange(1, self.num_frequencies + 1, device=x.device).float()
        x_expanded = x * frequencies.view(1, -1) * 2 * np.pi
        return torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=1)
