import torch

def save_model(model, path):
    '''Save the model state dictionary to the given path.'''
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device='cpu'):
    '''Instantiate model_class, load its state dictionary from path, and return the model.'''
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model
