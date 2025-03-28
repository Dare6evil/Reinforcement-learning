import gymnasium
import os
import sys
import torch

sys.path.append(os.pardir)
import modules

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
env = gymnasium.make('CartPole-v0', render_mode='human')
q = modules.Q(env.action_space.n, *env.observation_space.shape)
q.to(device)
q.eval()
q.load_state_dict(torch.load('DQN.pth', weights_only=True))
state, _ = env.reset()
while True:
    action = q(torch.Tensor(state).to(device)).argmax().item()
    next_state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
    state = next_state
env.close()
