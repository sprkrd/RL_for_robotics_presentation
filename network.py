import torch
import torch.nn as nn


class Network(nn.Module):
    
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        super().__init__()
        stack = []
        previous_layer_size = num_inputs
        for layer_size in hidden_layers:
            stack.append(nn.Linear(previous_layer_size, layer_size))
            stack.append(nn.ReLU())
            previous_layer_size = layer_size
        stack.append(nn.Linear(previous_layer_size, num_outputs))
        self.linear_relu_stack = nn.Sequential(*stack)
        
    def predict(self, x):
        x = torch.asarray(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.forward(x)
        
    def forward(self, x):
        return self.linear_relu_stack(x)
        
        
class PolicyNetwork(Network):
    
    def __init__(self, num_features, action_dim, hidden_layers):
        super().__init__(num_features, hidden_layers, action_dim)
        
        
class CriticNetwork(Network):
    
    def __init__(self, num_features, action_dim, hidden_layers):
        super().__init__(num_features+action_dim, hidden_layers, 1)
        
    def predict(self, state, action):
        state = torch.asarray(state, dtype=torch.float32)
        action = torch.asarray(action, dtype=torch.float32)
        x = torch.cat((state, action))
        return super().predict(x)
        

model = CriticNetwork(4, 2, (32,))
print(model.predict([1,2,3,4],[10,11]))

print(list(model.parameters()))
