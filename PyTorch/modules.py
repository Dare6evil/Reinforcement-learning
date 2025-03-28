import torch


class Policy(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        q = torch.nn.functional.relu(self.layer1(x))
        return torch.nn.functional.softmax(self.layer2(q), dim=0)


class Q(torch.nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, action_size)

    def forward(self, x):
        q = torch.nn.functional.relu(self.layer1(x))
        return self.layer2(q)


class V(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, 128)
        self.layer2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        v = torch.nn.functional.relu(self.layer1(x))
        return self.layer2(v)
