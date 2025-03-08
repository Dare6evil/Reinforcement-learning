import torch


class QNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, 100)
        self.layer2 = torch.nn.Linear(100, out_features)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        return self.layer2(x)
