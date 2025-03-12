import copy
import gymnasium
import numpy
import os
import sys

sys.path.append(os.pardir)
import data
import modules
import torch

annealing_num_steps = 1000
batch_size = 32
buffer_size = 10000
environment = gymnasium.make('LunarLander-v3', render_mode='human')
epsilon_end = 0.1
epsilon_init = 1.0
epsilon = epsilon_init
gamma = 0.9
learning_rate = 0.01
num_episodes = 300
observation, _ = environment.reset()
q = modules.Q(environment.action_space.n, *environment.observation_space.shape)
optimizer = torch.optim.SGD(q.parameters(), learning_rate)
replay_buffer = data.ReplayBuffer(buffer_size)
synchronization_interval = 20
target_q = modules.Q(environment.action_space.n, *environment.observation_space.shape)
for i in range(num_episodes):
    while True:
        if epsilon < numpy.random.rand():
            action = q(torch.Tensor(observation)).argmax().item()
        else:
            action = numpy.random.choice(environment.action_space.n)
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        replay_buffer.add(observation, action, reward, next_observation, int(done),
                          abs((1 - int(done)) * gamma * q(torch.Tensor(next_observation).detach()).max() + reward -
                              q(torch.Tensor(observation))[action]))
        if batch_size < len(replay_buffer):
            observation_batch, action_batch, reward_batch, next_observation_batch, done_batch, _ = replay_buffer.sample(
                batch_size)
            loss = torch.nn.functional.mse_loss((1 - torch.Tensor(numpy.array(done_batch))) * gamma * target_q(
                torch.Tensor(numpy.array(next_observation_batch)).detach()).max(axis=1).values + torch.Tensor(
                reward_batch), q(torch.Tensor(numpy.array(observation_batch)))[numpy.arange(batch_size), action_batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            observation, _ = environment.reset()
            break
        observation = next_observation
    epsilon -= (epsilon_init - epsilon_end) / annealing_num_steps
    if not i % synchronization_interval:
        target_q = copy.deepcopy(q)
environment.close()
torch.save(q.state_dict(), 'Deep Q-learning.pth')
