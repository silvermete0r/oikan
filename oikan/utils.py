from .exceptions import BasisError, DataTypeError
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline

class BSplineBasis(nn.Module):
    '''Module to compute B-Spline basis values for input features.'''
    def __init__(self, num_knots=10, degree=3):
        super().__init__()
        if num_knots < degree + 5:
            raise BasisError(f"Number of knots ({num_knots}) must be at least {degree + 5} for degree {degree}")
        self.num_knots = max(num_knots, degree + 5)
        self.degree = degree
        inner_knots = np.linspace(0, 1, self.num_knots - 2 * degree)
        left_pad = np.zeros(degree)
        right_pad = np.ones(degree)
        knots = np.concatenate([left_pad, inner_knots, right_pad])
        self.register_buffer('knots', torch.FloatTensor(knots))
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise DataTypeError("Input must be a torch.Tensor")
        try:
            x_np = x.detach().cpu().numpy()
            basis_values = np.zeros((x_np.shape[0], self.num_knots - self.degree - 1))
            x_min, x_max = x_np.min(), x_np.max()
            x_normalized = (x_np - x_min) / (x_max - x_min + 1e-8)
            for i in range(self.num_knots - self.degree - 1):
                spl = BSpline.basis_element(self.knots[i:i+self.degree+2])
                basis_values[:, i] = spl(x_normalized.squeeze())
            basis_values = np.nan_to_num(basis_values, 0)
            return torch.FloatTensor(basis_values).to(x.device)
        except Exception as e:
            raise BasisError(f"BSpline basis computation failed: {str(e)}")

class FourierBasis(nn.Module):
    '''Module to compute Fourier basis representations for input features.'''
    def __init__(self, num_frequencies=5):
        super().__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise DataTypeError("Input must be a torch.Tensor")
        try:
            frequencies = torch.arange(1, self.num_frequencies + 1, device=x.device, dtype=torch.float)
            x_expanded = x * frequencies.view(1, -1) * 2 * np.pi
            return torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=1)
        except Exception as e:
            raise BasisError(f"Fourier basis computation failed: {str(e)}")
