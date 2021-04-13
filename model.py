import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, ch_in=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ch_in, 32, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(inplace=True),
                                   nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(inplace=True),
                                   nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(inplace=True),
                                   nn.BatchNorm2d(64))
        self.fc1 = nn.Linear(7 * 7 * 64, 2)
        self.fc2 = nn.Linear(7 * 7 * 64, 2)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = x.view(-1, 7 * 7 * 64)

        mu = self.fc1(x)
        log_var = self.fc2(x)

        epsilon = torch.randn_like(mu)
        output = mu + torch.exp(log_var / 2) * epsilon

        return output, mu, log_var


class Decoder(nn.Module):
    def __init__(self, ch_in=2):
        super().__init__()
        self.fc = nn.Linear(ch_in, 7 * 7 * 64)
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(64))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x


class VAE(nn.Module):
    def __init__(self, ch_in=1):
        super().__init__()
        self.encoder = Encoder(ch_in=ch_in)
        self.decoder = Decoder(ch_in=2)

    def forward(self, x):
        x, mu, log_var = self.encoder(x)
        x = x.float()
        x = self.decoder(x)

        return x, mu, log_var