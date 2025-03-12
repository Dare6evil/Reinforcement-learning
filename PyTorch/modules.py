import torch


class Q(torch.nn.Module):
    def __init__(self, action_space_n, observation_space_n):
        super().__init__()
        self.layer1 = torch.nn.Linear(observation_space_n, 100)
        self.layer2 = torch.nn.Linear(100, action_space_n)

    def forward(self, state):
        q = torch.nn.functional.relu(self.layer1(state))
        return self.layer2(q)


class DuelingQNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, 100)
        self.layer2 = torch.nn.Linear(100, 1)
        self.layer3 = torch.nn.Linear(in_features, 100)
        self.layer4 = torch.nn.Linear(100, out_features)

    def forward(self, x):
        V = torch.nn.functional.relu(self.layer1(x))
        V = self.layer2(V)
        A = torch.nn.functional.relu(self.layer3(x))
        A = self.layer4(A)
        if 1 < A.ndim:
            return V + A - A.mean(axis=1, keepdim=True)
        return V + A - A.mean()
