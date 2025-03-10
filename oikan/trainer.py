import torch
import torch.nn as nn
from .regularization import RegularizedLoss

def train(model, train_data, epochs=100, lr=0.01, save_path=None, verbose=True):
    """Train regression model using MSE loss with regularization."""
    X_train, y_train = train_data
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
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def train_classification(model, train_data, epochs=100, lr=0.01, save_path=None, verbose=True):
    """Train classification model using CrossEntropy loss with regularization."""
    X_train, y_train = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
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
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
