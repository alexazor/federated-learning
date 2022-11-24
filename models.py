import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, nb_input: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nb_input, 1)
        # self.fc2 = nn.Linear(2, 1)
        # self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x