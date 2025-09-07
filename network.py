from typing import Self
import torch
import torch.nn as nn

from copy import deepcopy


class Network(nn.Module):
    
    def __init__(self, num_inputs: int, hidden_layers: tuple[int]|int, num_outputs: int):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = (hidden_layers,)
        stack = []
        previous_layer_size = num_inputs
        for layer_size in hidden_layers:
            stack.append(nn.Linear(previous_layer_size, layer_size))
            stack.append(nn.ReLU())
            previous_layer_size = layer_size
        stack.append(nn.Linear(previous_layer_size, num_outputs))
        self.linear_relu_stack = nn.Sequential(*stack)
        
    def forward(self, x) -> torch.Tensor:
        return self.linear_relu_stack(x)
    
    def clone(self, requires_grad=False) -> Self:
        clone = deepcopy(self)
        for param in clone.parameters():
            param.requires_grad = requires_grad
        return clone
        
        
class PolicyNetwork(Network):
    
    def __init__(self, num_features: int, action_dim: int, hidden_layers: tuple[int]|int):
        super().__init__(num_features, hidden_layers, action_dim)
        
        
class CriticNetwork(Network):
    
    def __init__(self, num_features:int, action_dim: int, hidden_layers: tuple[int]|int):
        super().__init__(num_features+action_dim, hidden_layers, 1)
        

# model = CriticNetwork(4, 2, (32,))
# print(model.predict([1,2,3,4],[10,11]))

# print(list(model.parameters()))
