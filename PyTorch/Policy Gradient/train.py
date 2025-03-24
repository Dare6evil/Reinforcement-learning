import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import modules

env = gymnasium.make('LunarLander-v3', render_mode='human')
gamma = 0.98
lr = 0.0002
num_episodes = 3000
pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
optimizer = torch.optim.SGD(pi.parameters(), lr)
state, _ = env.reset()
for _ in range(num_episodes):
    g = 0
    loss = 0
    memory = []
    while True:
        probs = pi(torch.Tensor(state))
        action = numpy.random.choice(len(probs), p=probs.detach().numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        memory.append([probs[action], reward])
        if terminated or truncated:
            state, _ = env.reset()
            break
        state = next_state
    for _, reward in reversed(memory):
        g = g * gamma + reward
    for prob, _ in reversed(memory):
        loss += -g * torch.log(prob)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
env.close()
torch.save(pi.state_dict(), 'Policy Gradient.pth')
