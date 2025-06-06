import os
import sys

sys.path.append(os.pardir)
import gymnasium
import modules
import numpy
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
pi.eval()
pi.load_state_dict(torch.load("REINFORCE.pth", weights_only=True))
pi.to(device)
state, _ = env.reset()
total_reward = 0
while True:
    action = numpy.random.choice(
        env.action_space.n, p=pi(torch.Tensor(state).to(device)).detach().cpu().numpy()
    )
    next_state, reward, terminated, _, _ = env.step(action)
    total_reward += reward
    if terminated:
        break
    state = next_state
env.close()
print(total_reward)
