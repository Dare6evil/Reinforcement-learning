import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import models

criterion = torch.nn.MSELoss()
environment = gymnasium.make('LunarLander-v3', render_mode='human')
epsilon = 0.1
gamma = 0.9
learning_rate = 0.01
observation, _ = environment.reset()
q_network = models.QNetwork(*environment.observation_space.shape, environment.action_space.n)
optimizer = torch.optim.SGD(q_network.parameters(), learning_rate)
for _ in range(1000):
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
environment.close()
torch.save(q_network.state_dict(), 'QNetwork.pth')
