from typing import Any, Sequence
from network import PolicyNetwork, CriticNetwork
from collections import namedtuple
import torch
import gymnasium as gym
import numpy as np
import random

class CircularBuffer[T]:
    def __init__(self, max_buffer_size: int):
        self._max_buffer_size = max_buffer_size
        self._offset = 0
        self._data = []
        
    def append(self, item: T) -> None:
        if len(self._data) < self._max_buffer_size:
            self._data.append(item)
        else:
            offset = self._offset
            self._data[offset] = item
            self._offset = (offset+1) % self._max_buffer_size
            
    def sample(self, k: int, replace: bool = True) -> Sequence[T]:
        k = min(k, len(self))
        return random.choices(self._data, k=k) if replace else random.sample(self._data, k=k)

    def __getitem__(self, index: int) -> T:
        return self._data[(self._offset + index)%len(self)]
        
    def __len__(self) -> int:
        return len(self._data)
        

Experience = namedtuple("Experience", "state action reward next_state done truncated")


class DDPG:
    
    DEFAULT_PARAMETERS = {
        "replay_buffer_size": 1000000,
        "gamma": 0.995,
        "number_of_epochs": 100,
        "steps_per_epoch": 1000,
        "batch_size": 100,
        "warmup_steps": 1000,
        "update_every": 50,
        "polyak": 0.995,
        "lr": 0.001,
        "momentum": 0.9,
        "number_of_tests": 100,
        "seed": None,
        "episode_cb": None,
        "hidden_layers_critic": (512,),
        "hidden_layers_actor": (512,),
        "action_noise": 0.1
    }
    
    def __init__(self, env: gym.Env, **parameters):
        self._env = env
        self._parameters = {**DDPG.DEFAULT_PARAMETERS, **parameters}
        seed = self._parameters["seed"]
        env.action_space.seed(seed)
        self._s, self._info = env.reset(seed=seed)
        self._replay_buffer = CircularBuffer[Experience](self._parameters["replay_buffer_size"])
        self._actor_network = PolicyNetwork(env.observation_space.shape[0],
                                            env.action_space.shape[0],
                                            self._parameters["hidden_layers_actor"])
        self._critic_network = CriticNetwork(env.observation_space.shape[0],
                                             env.action_space.shape[0],
                                             self._parameters["hidden_layers_critic"])
        self._target_actor_network = self._actor_network.clone()
        self._target_critic_network = self._critic_network.clone()
        self._optimizer_actor = torch.optim.SGD(self._actor_network.parameters(),
                                                lr=self._parameters["lr"],
                                                momentum=self._parameters["momentum"],
                                                maximize=True)
        self._optimizer_critic = torch.optim.SGD(self._critic_network.parameters(),
                                                 lr=self._parameters["lr"],
                                                 momentum=self._parameters["momentum"])
        
    def _add_experience(self, s: np.ndarray, a: np.ndarray, r: float, s_next: np.ndarray, done: bool, truncated: bool):
        self._replay_buffer.append(Experience(s,a,r,s_next,done,truncated))

    def get_last_episode(self) -> Sequence[Experience]:
        episode = []
        if self._replay_buffer:
            i = len(self._replay_buffer)-1
            episode.append(self._replay_buffer[i])
            i -= 1
            experience = self._replay_buffer[i]
            while i >= 0 and not experience.done and not experience.truncated:
                episode.append(experience)
                i -= 1
                experience = self._replay_buffer[i]
        episode.reverse()
        return episode
    
    def _step(self, action):
        episode_cb = self._parameters["episode_cb"]
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        s_next, r, done, truncated, info_next = self._env.step(action)
        self._add_experience(self._s, action, r, s_next, done, truncated)
        if done or truncated:
            self._s, self._info = self._env.reset()
            if episode_cb:
                episode_cb(self)
        else:
            self._s, self._info = s_next, info_next
        
    def _warmup(self, verbose: bool):
        if verbose:
            print("Warm-up steps before executing and updating target policy")
        for _ in range(self._parameters["warmup_steps"]):
            a = self._env.action_space.sample()
            self._step(a)

    def _update_polyak(self, source: torch.nn.Module, target: torch.nn.Module):
        polyak = self._parameters["polyak"]
        for weights_source, weights_target in zip(source.parameters(), target.parameters()):
            weights_target += (1 - polyak)*(weights_source - weights_target).detach()

    def _get_target(self, batch: Sequence[Experience]) -> torch.Tensor:
        gamma = self._parameters["gamma"]
        r = torch.Tensor([experience.reward for experience in batch])
        d = torch.Tensor([1.0*experience.done for experience in batch])
        next_states = np.array([experience.next_state for experience in batch])
        Qtarget = self._get_state_value(next_states, target=True).T.squeeze(0)
        target = r + gamma*(1-d)*Qtarget
        return target
    
    

    def _update_once(self):
        batch = self._replay_buffer.sample(self._parameters["batch_size"], False)
        states = np.array([experience.state for experience in batch])
        actions = np.array([experience.action for experience in batch])
        Q = self._get_Q(states, actions, target=False).T.squeeze(0)
        target = self._get_target(batch)
        loss = torch.mean((Q - target)**2)
        self._actor_network.requires_grad_(False)
        loss.backward()
        self._optimizer_critic.step()
        self._optimizer_critic.zero_grad()
        self._actor_network.requires_grad_(True)
        self._critic_network.requires_grad_(False)
        Qoptim = self._get_state_value(states, target=False)
        Qavg = torch.mean(Qoptim)
        Qavg.backward()
        self._optimizer_actor.step()
        self._optimizer_actor.zero_grad()
        self._critic_network.requires_grad_(True)
        self._update_polyak(self._actor_network, self._target_actor_network)
        self._update_polyak(self._critic_network, self._target_critic_network)

    def _update(self):
        for _ in range(self._parameters["update_every"]):
            self._update_once()

    def _get_action(self, state, target=False, add_noise=False):
        actor = self._target_actor_network if target else self._actor_network

        state = torch.as_tensor(state).to(torch.float32)
        single_action = state.ndim == 1
        if single_action:
            state = state.unsqueeze(0)

        action = actor.forward(state)
        if add_noise:
            action += torch.randn_like(action)*self._parameters["action_noise"]

        if single_action:
            action = action.squeeze(0)
        return action
    
    def _get_Q(self, state, action, target=True) -> torch.Tensor:
        critic = self._target_critic_network if target else self._critic_network

        state = torch.as_tensor(state).to(torch.float32)
        single_value = state.ndim == 1
        if single_value:
            state = state.unsqueeze(0)

        action = torch.as_tensor(action).to(torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        state_action = torch.hstack((state, action))
        value = critic.forward(state_action)

        if single_value:
            value = value.squeeze()
        return value
    
    def _get_state_value(self, state, target=True):
        state = torch.as_tensor(state).to(torch.float32)
        action = self._get_action(state, target=target, add_noise=False)
        return self._get_Q(state, action, target=target)
    
    def _test(self):
        number_of_tests = self._parameters["number_of_tests"]
        avg_completion = 0
        avg_reward = 0
        for _ in range(number_of_tests):
            s, _ = self._env.reset()
            done, truncated = False, False
            while not done and not truncated:
                action = self._get_action(s, target=False, add_noise=False).detach().numpy()
                s_next, r, done, truncated, _ = self._env.step(action)
                avg_reward += r
                s = s_next
            avg_completion += done
        avg_completion /= number_of_tests
        avg_reward /= number_of_tests
        return avg_completion, avg_reward
    
    def train(self, verbose: bool = False):
        if self._parameters["warmup_steps"] > 0:
            self._warmup(verbose)
        steps_before_update = self._parameters["update_every"]
        for epoch in range(1, 1+self._parameters["number_of_epochs"]):
            self._s, self._info = self._env.reset()
            if verbose:
                print(f"Starting epoch #{epoch}. ", end="", flush=True)
            for _ in range(self._parameters["steps_per_epoch"]):
                a = self._get_action(self._s, target=False, add_noise=True)
                self._step(a)
                steps_before_update -= 1
                if steps_before_update == 0:
                    self._update()
                    steps_before_update = self._parameters["update_every"]
            if self._parameters["number_of_tests"] > 0:
                avg_completion, avg_reward = self._test()
                print(f"Avg. completion: {avg_completion:.1%}, avg. reward: {avg_reward:.3}")
            else:
                print("Epoch finished. No tests done")
                
