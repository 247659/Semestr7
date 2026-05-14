from torch import nn


class ImagenetteCNN2FeaturesSecond(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.feature_layer = nn.Sequential(

            nn.Linear(64 * 20 * 20, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

        self.classifier = nn.Linear(2, 10)

    def forward(self, x):

        x = self.features(x)

        x = self.flatten(x)

        features = self.feature_layer(x)

        output = self.classifier(features)

        return output, features