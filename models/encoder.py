from torch import nn
from torchsummary import summary


class ConditionalEncoder(nn.Module):
    """Basic CNN without the fully connected layers on the end"""

    def __init__(self, num_channels, image_size, z_dim):
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
            nn.Linear(64 * int(image_size / 16) * int(image_size / 16), 500),
            # Multiply output dimension by 2, so that 0-z can parameterize mu and z-2z can parametereize log(var)
            nn.Linear(500, z_dim * 2),
        )

    def forward(self, X):
        return self.model(X)


class ConditionalEncoderAlt(nn.Module):
    """Basic CNN without the fully connected layers on the end"""

    def __init__(self, num_channels, image_size, z_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 8, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(int(image_size / 2) * int(image_size / 2), 300),
            nn.Linear(300, z_dim * 2),
        )

    def forward(self, X):
        return self.model(X)


if __name__ == "__main__":
    img_size = 32
    num_channels = 1
    z_dim = 50
    summary(
        ConditionalEncoderAlt(num_channels, img_size, z_dim).model,
        (num_channels, img_size, img_size),
    )
