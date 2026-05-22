import torch
import torch.nn as nn


class CNN2Features(nn.Module):

    def __init__(self):
        super(CNN2Features, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(32 * 7 * 7, 2)
        self.classifier = nn.Linear(2, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        features = self.feature_layer(x)
        output = self.classifier(features)

        return output, features