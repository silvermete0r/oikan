import torch.nn as nn

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
        Activation function ('relu' or 'tanh').
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
            else:
                raise ValueError("Unsupported activation function.")
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)