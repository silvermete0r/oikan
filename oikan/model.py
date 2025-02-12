import torch
import torch.nn as nn

# EfficientKAN Layer
class EfficientKAN(nn.Module):
    def __init__(self, input_dim, hidden_units=10):
        super(EfficientKAN, self).__init__()
        self.basis_functions = nn.ModuleList([nn.Linear(1, hidden_units) for _ in range(input_dim)])
        self.activations = nn.ReLU()
    
    def forward(self, x):
        transformed_features = [self.activations(bf(x[:, i].unsqueeze(1))) for i, bf in enumerate(self.basis_functions)]
        return torch.cat(transformed_features, dim=1)

# OIKAN Model
class OIKAN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=10):
        super(OIKAN, self).__init__()
        self.efficientkan = EfficientKAN(input_dim, hidden_units)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * hidden_units, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        transformed_x = self.efficientkan(x)
        return self.mlp(transformed_x)