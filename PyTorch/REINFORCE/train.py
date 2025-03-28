from matplotlib import pyplot
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
num_episodes = 300
pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
optimizer_pi = torch.optim.SGD(pi.parameters(), lr)
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
state, _ = env.reset(seed=seed)
total_rewards = []
v = modules.V(*env.observation_space.shape)
optimizer_v = torch.optim.SGD(v.parameters(), lr)
for _ in range(num_episodes):
    g = 0
    loss_pi = 0
    loss_v = 0
    memory = []
    total_reward = 0
    while True:
        b = v(torch.Tensor(state))
        probs = pi(torch.Tensor(state))
        action = numpy.random.choice(len(probs), p=probs.detach().numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        loss_v += torch.nn.functional.mse_loss((1 - done) * gamma * v(torch.Tensor(next_state).detach()) + reward, b)
        memory.append([b, probs[action], reward])
        total_reward += reward
        if done:
            state, _ = env.reset()
            break
        state = next_state
    for b, prob, reward in reversed(memory):
        g = g * gamma + reward
        loss_pi += -(g - b) * torch.log(prob)
    optimizer_pi.zero_grad()
    loss_pi.backward(retain_graph=True)
    optimizer_v.zero_grad()
    loss_v.backward()
    optimizer_pi.step()
    optimizer_v.step()
    total_rewards.append(total_reward)
env.close()
pyplot.plot(total_rewards)
pyplot.show()
# torch.save(pi.state_dict(), 'REINFORCE.pth')
