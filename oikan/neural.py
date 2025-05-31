import torch.nn as nn
import torch

class TabularNet(nn.Module):
    """
    Feedforward neural network for tabular data.
    
    Parameters:
    -----------
    input_size : int
        Number of input features.
    hidden_sizes : list
        List of hidden layer sizes.
    output_size : int
        Number of output units.
    activation : str, optional (default='relu')
        Activation function ('relu', 'tanh', 'leaky_relu', 'elu', 'swish', 'gelu').
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(TabularNet, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'elu':
                layers.append(nn.ELU(alpha=1.0))
            elif activation == 'swish':
                layers.append(nn.SiLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError("Unsupported activation function.")
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)