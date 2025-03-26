import torch
import torch.nn as nn

class RegularizedLoss:
    def __init__(self, base_criterion, model, l1_lambda=0.01, gradient_lambda=0.01):
        self.base_criterion = base_criterion  # Primary loss (e.g. MSE, CrossEntropy)
        self.model = model
        self.l1_lambda = l1_lambda
        self.gradient_lambda = gradient_lambda
    
    def __call__(self, pred, target, inputs):
        # Compute the standard loss
        base_loss = self.base_criterion(pred, target)
        
        # Calculate L1 regularization to promote sparsity
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.norm(param, p=1)
            
        # Compute gradient penalty to enforce smoothness
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        
        # For classification, take mean over all output dimensions
        if len(outputs.shape) > 1:
            outputs = outputs.mean(dim=1)
        
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=inputs,
            create_graph=True,
            allow_unused=True
        )[0]
        
        if gradients is not None:
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        else:
            grad_penalty = 0
        
        return base_loss + self.l1_lambda * l1_loss + self.gradient_lambda * grad_penalty
