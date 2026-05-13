import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self, num_classes=10, in_channels=1):
        super(CNNClassifier, self).__init__()

        # in_channels to kanały wejściowe 1 dlatego, że MNIST to obrazy w skali szarości,
        # wiec dla mnist mamy 1 x 28 x 28
        # out_channels to liczba filtrów / map cech - przy czym kazdy filtr szuka czegos innego
        # tworzy sie nam coś takiego 32 x 28 x 28
        # kernel_size - rozmiar filtra, który przesuwa się po obrazie
        # poling zmiejsza rozmiar obrazu - wartość dwa czyli, 32x28x28 -> 32x14x14
        # padding jest po to zeby filtr mogł pokryć takze brzegi obrazu
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x