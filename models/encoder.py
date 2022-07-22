from torch import nn
from torchsummary import summary


class Encoder(nn.Module):
    """Basic CNN without the fully connected layers on the end"""

    def __init__(self, num_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                num_channels, 32, 4, 2, 1
            ),  # B,  32, input_size / 2, input_size / 2
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, input_size / 4, input_size / 4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64, input_size / 8, input_size / 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64, input_size / 16, input_size / 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),  # B, 64 * (input_size / 16) * (input_size / 16)
        )

    def forward(self, X):
        return self.model(X)


if __name__ == "__main__":
    img_size = 32
    num_channels = 1
    summary(Encoder(num_channels).model, (num_channels, img_size, img_size))
