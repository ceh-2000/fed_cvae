from torch import nn
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, num_channels, image_size):
        super().__init__()

        assert image_size % 16 == 0, "Input size must be divisible by 16"

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),  # B,  32, input_size / 2, input_size / 2
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
    summary(Encoder(1, img_size).model, (1, img_size, img_size))
