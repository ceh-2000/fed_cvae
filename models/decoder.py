import torch
from torch import nn
from torchsummary import summary

from models.view import View


class ConditionalDecoder(nn.Module):
    """
    Decoder needs to take in a sampled z AND the true class passed from the forward method
    """

    def __init__(self, image_size, num_classes, num_channels, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                z_dim + num_classes, 64 * int(image_size / 16) * int(image_size / 16)
            ),  # B, 64 * (input_size / 16) * (input_size / 16)
            View(
                (-1, 64, int(image_size / 16), int(image_size / 16))
            ),  # B, 64, input_size / 16, input_size / 16
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 64, 4, 2, 1
            ),  # B,  64, input_size / 8, input_size / 8
            nn.BatchNorm2d(64, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),  # B,  32, input_size / 4, input_size / 4
            nn.BatchNorm2d(32, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 32, 4, 2, 1
            ),  # B,  32, input_size / 2, input_size / 2
            nn.BatchNorm2d(32, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, num_channels, 4, 2, 1
            ),  # B, nc, input_size, input_size
        )

    def forward(self, z, y_hot):
        y_hot = y_hot.to(torch.float32)
        concat_vec = torch.cat((z, y_hot), 1)  # sampled z and a one-hot class vector

        return self.model(concat_vec)


if __name__ == "__main__":
    img_size = 32
    num_classes = 10
    num_channels = 1
    z_dim = 50
    summary(
        ConditionalDecoder(img_size, num_classes, num_channels, z_dim).model,
        (1, (z_dim + num_classes)),
    )
