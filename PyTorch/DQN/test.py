import gymnasium
import os
import sys
import torch

sys.path.append(os.pardir)
import modules

env = gymnasium.make('LunarLander-v3', render_mode='human')
q = modules.Q(env.action_space.n, *env.observation_space.shape)
q.eval()
q.load_state_dict(torch.load('DQN.pth', weights_only=True))
state, _ = env.reset()
while True:
    action = q(torch.Tensor(state)).argmax().item()
    next_state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state
env.close()
