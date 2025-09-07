import gymnasium as gym
import random
import torch
import numpy as np

from ddpg import DDPG

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=.1)

agent_parameters = {
    "hidden_layers_critic": (32,),
    "hidden_layers_actor": (32,),
    "steps_per_epoch": 100000,
    "warmup_steps": 100000,
    "seed": SEED,
    "action_noise": 0.2
}

agent = DDPG(env, **agent_parameters)
agent.train(verbose=True)

