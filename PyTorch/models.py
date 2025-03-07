import torch


class QNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.container = torch.nn.Sequential(
            torch.nn.Linear(in_features, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_features),
        )

    def forward(self, x):
        return self.container(x)


class DeepQNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
