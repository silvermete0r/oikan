import torch
import torch.nn as nn
from .regularization import RegularizedLoss

def train(model, train_data, epochs=100, lr=0.01):
    '''Train regression model using MSE loss with regularization.'''
    X_train, y_train = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    reg_loss = RegularizedLoss(criterion, model)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_train)
        # Compute loss including regularization penalties
        loss = reg_loss(outputs, y_train, X_train)
        loss.backward()  # Backpropagate errors
        optimizer.step()  # Update parameters
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def train_classification(model, train_data, epochs=100, lr=0.01):
    '''Train classification model using CrossEntropy loss with regularization.'''
    X_train, y_train = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    reg_loss = RegularizedLoss(criterion, model)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients each epoch
        outputs = model(X_train)
        # Loss includes both cross-entropy and regularization terms
        loss = reg_loss(outputs, y_train, X_train)
        loss.backward()  # Backpropagation
        optimizer.step()  # Parameter update
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
