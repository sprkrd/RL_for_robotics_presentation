from network import PolicyNetwork, CriticNetwork
import torch

torch.manual_seed(42)

number_of_state_features = 4
action_dims = 2

policy_network = PolicyNetwork(number_of_state_features, action_dims, (256,))
critic_network = CriticNetwork(number_of_state_features, action_dims, (256,))

state = [1,2,3,4]

state = torch.asarray(state, dtype=torch.float32).unsqueeze(0)
print(state)

for p in critic_network.parameters():
    p.requires_grad = False

optimizer = torch.optim.SGD(policy_network.parameters(), lr=1e-3, momentum=0.9, maximize=True)
for _ in range(200):
    action = policy_network.forward(state)
    state_action = torch.hstack((state,action))
    action_value = - torch.mean(critic_network.forward(state_action))
    print(action_value)
    action_value.backward()
    optimizer.step()
    optimizer.zero_grad()

# print(action_value)

