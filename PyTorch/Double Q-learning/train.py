import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import data
import modules

annealing_num_steps = 1000
batch_size = 32
buffer_size = 10000
env = gymnasium.make('LunarLander-v3', render_mode='human')
eps_end = 0.1
eps_init = 1.0
eps = eps_init
gamma = 0.9
lr = 0.01
num_episodes = 300
q = modules.Q(*env.observation_space.shape, env.action_space.n)
optimizer = torch.optim.SGD(q.parameters(), lr)
replay_buffer = data.ReplayBuffer(buffer_size)
state, _ = env.reset()
sync_interval = 20
target_q = modules.Q(*env.observation_space.shape, env.action_space.n)
for i in range(num_episodes):
    while True:
        if eps < numpy.random.rand():
            action = q(torch.Tensor(state)).argmax().item()
        else:
            action = numpy.random.choice(env.action_space.n)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, int(done),
                          abs((1 - int(done)) * gamma * q(torch.Tensor(next_state).detach()).max() + reward -
                              q(torch.Tensor(state))[action]))
        if batch_size < len(replay_buffer):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, _ = replay_buffer.sample(batch_size)
            loss = torch.nn.functional.mse_loss((1 - torch.Tensor(numpy.array(done_batch))) * gamma *
                                                target_q(torch.Tensor(numpy.array(next_state_batch)).detach())[
                                                    numpy.arange(batch_size), q(
                                                        torch.Tensor(numpy.array(state_batch))).argmax(
                                                        dim=1)] + torch.Tensor(reward_batch),
                                                q(torch.Tensor(numpy.array(state_batch)))[
                                                    numpy.arange(batch_size), action_batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            state, _ = env.reset()
            break
        state = next_state
    eps = max(eps - (eps_init - eps_end) / annealing_num_steps, eps_end)
    if not i % sync_interval:
        target_q.load_state_dict(q.state_dict())
env.close()
torch.save(q.state_dict(), 'Double Q-learning.pth')
