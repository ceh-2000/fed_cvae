import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from torch import nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from models.decoder import (ConditionalDecoder, ConditionalDecoderAlt,
                            ConditionalDecoderResNet)
from models.encoder import (ConditionalEncoder, ConditionalEncoderAlt,
                            ConditionalEncoderResNet)


class CVAE(nn.Module):
    """
    A slight modification of the model proposed in original Beta-VAE paper (Higgins et al, ICLR, 2017).

    Compatible with input images that are of spatial dimension divisible by 16, includes a classifier as a component
    of the pipeline, and allows image generation conditional on a chosen class.
    """

    def __init__(self, num_classes, num_channels, z_dim, image_size, version):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.z_dim = z_dim

        if image_size % 16 != 0:
            raise Exception("Image size must be divisible by 16")

        self.image_size = image_size

        # Latent dist for further sampling: multivariate normal, z ~ N(0, I)
        self.mvn_dist = MultivariateNormal(
            torch.zeros(self.z_dim), torch.eye(self.z_dim)
        )

        # Define neural models needed for this implementation
        if version == 0:
            print("Standard model")
            self.encoder = ConditionalEncoder(
                num_channels=self.num_channels,
                image_size=self.image_size,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoder(
                image_size=self.image_size,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        elif version == 1:
            print("Alt model")
            self.encoder = ConditionalEncoderAlt(
                num_channels=self.num_channels,
                image_size=self.image_size,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoderAlt(
                image_size=self.image_size,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        elif version == 2:
            print("ResNet Model")
            self.encoder = ConditionalEncoderResNet(
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoderResNet(
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        else:
            raise NotImplementedError(
                "The model you specified has not been implemented."
            )

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                self.kaiming_init(m)

    def reparametrize(self, mu, logvar):
        """Re-paramaterization trick to make our CVAE fully-differentiable"""
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def sample_z(self, num_samples, dist, width=(-1, 1)):
        """Sample latent vectors"""
        if dist == "mvn":  # multivariate normal
            z = self.mvn_dist.sample((num_samples,))
        elif dist == "truncnorm":  # truncated multivariate normal
            truncnorm_tensor = torch.FloatTensor(
                truncnorm.rvs(a=width[0], b=width[1], size=num_samples * self.z_dim)
            )
            z = torch.reshape(truncnorm_tensor, (num_samples, self.z_dim))
        elif dist == "uniform":  # uniform
            z = torch.FloatTensor(num_samples, self.z_dim).uniform_(*width)

        else:
            raise NotImplementedError(
                "Only multivariate normal (mvn), truncated multivariate normal (truncnorm), and uniform (uniform) distributions supported."
            )

        return z

    def forward(self, X, y_hot, device):
        distributions = self.encoder(X)

        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]

        # Re-paramaterization trick, sample latent vector z
        z = self.reparametrize(mu, logvar).to(device)

        # Decode latent vector + class info into a reconstructed image
        x_recon = self.decoder(z, y_hot)

        return x_recon, mu, logvar


if __name__ == "__main__":
    num_samples = 500000
    z_dim = 10
    std = 1

    # Uniform example
    uniform_width = (-1 * std, std)
    uniform_tensor = torch.FloatTensor(num_samples, z_dim).uniform_(*uniform_width)
    print(uniform_tensor.shape)

    # Visual check
    sns.histplot(data=uniform_tensor[:, 0])
    plt.show()

    # Truncated normal example
    truncnorm_tensor = torch.tensor(
        truncnorm.rvs(a=-1 * std, b=std, size=num_samples * z_dim)
    )
    truncnorm_tensor = torch.reshape(truncnorm_tensor, (num_samples, z_dim))
    print(truncnorm_tensor.shape)

    # Visual check
    sns.histplot(data=truncnorm_tensor[:, 0])
    plt.show()

    # # Create new instance of model
    # img_size = 32
    # num_classes = 10
    # num_channels = 1
    # z_dim = 50
    # version = 1
    #
    # model = CVAE(num_classes, num_channels, z_dim, img_size, version)
    #
    # # Generate dummy data
    # X = torch.rand((2, 1, 32, 32))
    # y_hot = torch.Tensor(
    #     [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    # )
    #
    # print("X:", X.shape)
    # print("y one-hot-encoded:", y_hot.shape)
    # print()
    #
    # # Run model forward pass
    # x_recon, mu, logvar = model(X, y_hot)
    # print("X reconstruction:", x_recon.shape)
    # print("mu:", mu.shape)
    # print("logvar:", logvar.shape)
