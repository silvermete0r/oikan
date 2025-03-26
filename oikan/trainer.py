import torch
import torch.nn as nn
from .regularization import RegularizedLoss

def train(model, train_data, epochs=100, lr=0.01, verbose=True):
    """Train regression model using MSE loss with regularization."""
    X_train, y_train = train_data
    
    # Convert inputs to tensors and ensure correct shape
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    
    # Move to device
    X_train = X_train.to(model.device)
    y_train = y_train.to(model.device)
    
    # Ensure y_train has shape (n_samples, output_dim)
    if len(y_train.shape) == 1:
        y_train = y_train.view(-1, 1)
    
    # Initialize model if not already initialized
    if not model.is_initialized:
        model.initialize_from_data(X_train, y_train)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    reg_loss = RegularizedLoss(criterion, model)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = reg_loss(outputs, y_train, X_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 and verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def train_classification(model, train_data, epochs=100, lr=0.01, verbose=True):
    """Train classification model using appropriate loss with regularization."""
    X_train, y_train = train_data
    
    # Convert and move to device
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)
    
    X_train = X_train.to(model.device)
    y_train = y_train.to(model.device)
    
    # Initialize model if not already initialized
    if not model.is_initialized:
        model.initialize_from_data(X_train, y_train)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multi-class
    if model.output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()
        # Convert labels to float for binary classification
        y_train = y_train.float()
    else:
        criterion = nn.CrossEntropyLoss()
    
    reg_loss = RegularizedLoss(criterion, model)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        
        # Handle output shape for binary classification
        if model.output_dim == 1:
            outputs = outputs.view(-1)
            
        loss = reg_loss(outputs, y_train, X_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 and verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model
