import copy
import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import data
import models

batch_size = 32
buffer_size = 10000
criterion = torch.nn.MSELoss()
environment = gymnasium.make('LunarLander-v3', render_mode='human')
epsilon_decay = 0.995
epsilon_end = 0.1
epsilon_start = 1.0
epsilon = epsilon_start
gamma = 0.9
learning_rate = 0.01
n_episodes = 1000
observation, _ = environment.reset()
q_network = models.QNetwork(*environment.observation_space.shape, environment.action_space.n)
optimizer = torch.optim.SGD(q_network.parameters(), learning_rate)
replay_buffer = data.ReplayBuffer(buffer_size)
synchronization_cycle = 20
target_q_network = models.QNetwork(*environment.observation_space.shape, environment.action_space.n)
for n_episode in range(n_episodes):
    g = 0
    while True:
        if epsilon < numpy.random.rand():
            action = q_network(torch.Tensor(observation)).argmax().item()
        else:
            action = numpy.random.choice(environment.action_space.n)
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        replay_buffer.add(observation, action, reward, next_observation, int(done))
        if batch_size < len(replay_buffer):
            observation_batch, action_batch, reward_batch, next_observation_batch, done_batch = replay_buffer.sample(
                batch_size)
            loss = criterion((1 - torch.Tensor(numpy.array(done_batch))) * gamma * target_q_network(
                torch.Tensor(numpy.array(next_observation_batch)).detach()).max(axis=1).values + torch.Tensor(
                reward_batch), q_network(torch.Tensor(numpy.array(observation_batch)))[
                                 numpy.arange(batch_size), action_batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            observation, _ = environment.reset()
            break
        observation = next_observation
    epsilon = max(epsilon * epsilon_decay, epsilon_end)
    if not n_episode % synchronization_cycle:
        target_q_network = copy.deepcopy(q_network)
environment.close()
torch.save(q_network.state_dict(), 'DeepQNetwork.pth')
