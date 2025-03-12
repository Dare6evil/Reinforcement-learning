import gymnasium
import os
import sys

sys.path.append(os.pardir)
import modules
import torch

environment = gymnasium.make('LunarLander-v3', render_mode='human')
observation, _ = environment.reset()
q = modules.Q(environment.action_space.n, *environment.observation_space.shape)
q.eval()
q.load_state_dict(torch.load('Deep Q-learning.pth', weights_only=True))
while True:
    action = q(torch.Tensor(observation)).argmax().item()
    next_observation, _, terminated, truncated, _ = environment.step(action)
    if terminated or truncated:
        break
    observation = next_observation
environment.close()
