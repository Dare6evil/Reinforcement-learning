import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import models

criterion = torch.nn.MSELoss()
environment = gymnasium.make('LunarLander-v3', render_mode='human')
episode = 1000
epsilon_decay = 0.995
epsilon_end = 0.1
epsilon_start = 1.0
epsilon = epsilon_start
gamma = 0.9
learning_rate = 0.01
observation, _ = environment.reset()
q_network = models.QNetwork(*environment.observation_space.shape, environment.action_space.n)
optimizer = torch.optim.SGD(q_network.parameters(), learning_rate)
for i in range(episode):
    while True:
        if epsilon < numpy.random.rand():
            action = q_network(torch.Tensor(observation)).argmax().item()
        else:
            action = numpy.random.choice(environment.action_space.n)
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        loss = criterion((1 - int(done)) * gamma * q_network(torch.Tensor(next_observation).detach()).max() + reward,
                         q_network(torch.Tensor(observation))[action])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if done:
            observation, _ = environment.reset()
            break
        observation = next_observation
    epsilon = max(epsilon * epsilon_decay, epsilon_end)
environment.close()
torch.save(q_network.state_dict(), 'QNetwork.pth')
