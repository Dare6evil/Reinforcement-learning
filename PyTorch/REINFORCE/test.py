import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import modules

env = gymnasium.make('LunarLander-v3', render_mode='human')
pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
pi.eval()
pi.load_state_dict(torch.load('REINFORCE.pth', weights_only=True))
state, _ = env.reset()
while True:
    action = numpy.random.choice(env.action_space.n, p=pi(torch.Tensor(state)).detach().numpy())
    next_state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state
env.close()
