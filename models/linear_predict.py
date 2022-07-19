from torch import nn
from torchsummary import summary


class LinearPredict(nn.Module):
    """To predict mu and logvar - 2 FC layers"""

    def __init__(self, image_size, z_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(64 * int(image_size / 16) * int(image_size / 16), 500),
            # Multiply output dimension by 2, so that 0-z can parameterize mu and z-2z can parametereize log(var)
            nn.Linear(500, z_dim * 2),
        )

    def forward(self, feature_rep):
        return self.model(feature_rep)


if __name__ == "__main__":
    img_size = 32
    z_dim = 50
    summary(
        LinearPredict(img_size, z_dim).model,
        (1, (64 * int(img_size / 16) * int(img_size / 16))),
    )
