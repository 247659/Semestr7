import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, use_relu=True):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.use_relu = use_relu

    def forward(self, x):
        x = self.fc1(x)

        if self.use_relu:
            x = F.relu(x)

        x = self.fc2(x)

        return x