import torch.optim as optim
import torch.nn as nn

# Regression training
def train(model, train_loader, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=lr)

    def closure():
        optimizer.zero_grad()
        outputs = model(train_loader[0])
        loss = criterion(outputs, train_loader[1])
        loss.backward()
        print(f"Loss: {loss.item()}")
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
        
# Classification training
def train_classification(model, train_loader, epochs=100, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_loader[0])
        loss = criterion(outputs, train_loader[1])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")