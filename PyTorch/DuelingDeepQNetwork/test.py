import gymnasium
import os
import sys
import torch

sys.path.append(os.pardir)
import models

environment = gymnasium.make('LunarLander-v3', render_mode='human')
observation, _ = environment.reset()
q_network = models.QNetwork(*environment.observation_space.shape, environment.action_space.n)
q_network.eval()
q_network.load_state_dict(torch.load('Deep Q-learning.pth', weights_only=True))
while True:
    action = q_network(torch.Tensor(observation)).argmax().item()
    next_observation, reward, terminated, truncated, _ = environment.step(action)
    done = terminated or truncated
    if done:
        break
    observation = next_observation
environment.close()
