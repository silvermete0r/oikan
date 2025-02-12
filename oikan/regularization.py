import torch
import torch.nn as nn

class RegularizedLoss:
    def __init__(self, base_criterion, model, l1_lambda=0.01, gradient_lambda=0.01):
        self.base_criterion = base_criterion
        self.model = model
        self.l1_lambda = l1_lambda
        self.gradient_lambda = gradient_lambda
    
    def __call__(self, pred, target, inputs):
        base_loss = self.base_criterion(pred, target)
        
        # L1 regularization
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.norm(param, p=1)
            
        # Gradient penalty
        grad_penalty = 0
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=inputs,
            create_graph=True
        )[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return base_loss + self.l1_lambda * l1_loss + self.gradient_lambda * grad_penalty
