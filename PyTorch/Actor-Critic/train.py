from matplotlib import pyplot
import gymnasium
import numpy
import os
import sys
import torch

sys.path.append(os.pardir)
import modules

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
episodes = 3000
gamma = 0.98
lr_pi = 0.0002
lr_v = 0.0005
reward_history = [0] * episodes
runs = 100
for run in range(1, 1 + runs):
    pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
    pi.to(device)
    optimizer_pi = torch.optim.Adam(pi.parameters(), lr_pi)
    state, _ = env.reset()
    v = modules.V(*env.observation_space.shape)
    v.to(device)
    optimizer_v = torch.optim.Adam(v.parameters(), lr_v)
    for episode in range(episodes):
        total_reward = 0
        while True:
            b = v(torch.Tensor(state).to(device))
            probs = pi(torch.Tensor(state).to(device))
            action = numpy.random.choice(len(probs), p=probs.detach().cpu().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            target = (1 - done) * gamma * v(torch.Tensor(next_state).to(device).detach()) + reward
            loss_pi = -(target - b) * torch.log(probs[action])
            optimizer_pi.zero_grad()
            loss_pi.backward(retain_graph=True)
            loss_v = torch.nn.functional.mse_loss(target, b)
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_pi.step()
            optimizer_v.step()
            total_reward += reward
            if done:
                state, _ = env.reset()
                break
            state = next_state
        reward_history[episode] += (total_reward - reward_history[episode]) / run
    # torch.save(pi.state_dict(), "Actor-Critic.pth")
env.close()
pyplot.plot(reward_history)
pyplot.show()
