import torch


class Q(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, 100)
        self.layer2 = torch.nn.Linear(100, out_features)

    def forward(self, x):
        q = torch.nn.functional.relu(self.layer1(x))
        return self.layer2(q)


class DuelingQ(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_features, 100)
        self.layer2 = torch.nn.Linear(100, 1)
        self.layer3 = torch.nn.Linear(in_features, 100)
        self.layer4 = torch.nn.Linear(100, out_features)

    def forward(self, x):
        v = torch.nn.functional.relu(self.layer1(x))
        v = self.layer2(v)
        a = torch.nn.functional.relu(self.layer3(x))
        a = self.layer4(a)
        if 1 < a.ndim:
            return v + a - a.mean(dim=1, keepdim=True)
        return v + a - a.mean()
