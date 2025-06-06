from matplotlib import pyplot
import os
import sys

sys.path.append(os.pardir)
import gymnasium
import modules
import numpy
import torch

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
env = gymnasium.make("CartPole-v1", render_mode="human")
episodes = 1000
gamma = 0.98
lr = 0.0002
# max_total_reward = 0
reward_history = [0] * episodes
runs = 5
for run in range(1, 1 + runs):
    pi = modules.Policy(env.action_space.n, *env.observation_space.shape)
    optimizer = torch.optim.Adam(pi.parameters(), lr)
    pi.to(device)
    state, _ = env.reset()
    for episode in range(episodes):
        g = 0
        loss = 0
        memory = []
        total_reward = 0
        while True:
            probs = pi(torch.Tensor(state).to(device))
            action = numpy.random.choice(len(probs), p=probs.detach().cpu().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append([probs[action], reward])
            total_reward += reward
            if done:
                state, _ = env.reset()
                break
            state = next_state
        for prob, reward in reversed(memory):
            g = g * gamma + reward
            loss += -g * torch.log(prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if max_total_reward < total_reward:
        #     max_total_reward = total_reward
        #     torch.save(pi.state_dict(), "REINFORCE.pth")
        # elif max_total_reward == total_reward:
        #     torch.save(pi.state_dict(), "REINFORCE.pth")
        reward_history[episode] += (total_reward - reward_history[episode]) / run
env.close()
pyplot.plot(reward_history)
pyplot.xlabel("Episode")
pyplot.ylabel("Total Reward")
pyplot.show()
