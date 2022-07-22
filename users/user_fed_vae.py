import copy

import torch.nn.functional as F
from torch.optim import Adam

from models.VAE import CVAE
from users.user import User
from utils import one_hot_encode


class UserFedVAE(User):
    def __init__(self, base_params, z_dim, image_size, beta):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.model = CVAE(
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            z_dim=z_dim,
            image_size=image_size,
        )
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.beta = beta

    def reconstruction_loss(self, x, x_recon):
        """Compute the reconstruction loss comparing input and reconstruction
        using an appropriate distribution given the number of channels.

        :param x: the input image
        :param x_recon: the reconstructed image produced by the decoder

        :return: reconstruction loss
        """

        batch_size = x.size(0)
        assert batch_size != 0

        # Use w/one-channel images
        if self.num_channels == 1:
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, reduction="sum"
            ).div(batch_size)
        # Multi-channel images
        elif self.num_channels == 3:
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, reduction="sum").div(batch_size)
        else:
            raise NotImplementedError("We only support 1 and 3 channel images.")

        return recon_loss

    def kl_divergence(self, mu, logvar):
        """Compute KL Divergence between the multivariate normal distribution of z
        and a multivariate standard normal distribution.

        :param mu: the mean of the predicted distribution
        :param logvar: the log-variance of the predicted distribution

        :return: total KL divergence loss
        """

        batch_size = mu.size(0)
        assert batch_size != 0

        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        # Shortcut: KL divergence w/N(0, I) prior and encoder dist is a multivariate normal
        # Push from multivariate normal --> multivariate STANDARD normal X ~ N(0,I)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)

        return total_kld

    def train(self, local_epochs):
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):

                y_hot = one_hot_encode(y_batch, self.num_classes)

                X_recon, mu, logvar = self.model(X_batch, y_hot)

                # Calculate losses
                recon_loss = self.reconstruction_loss(X_batch, X_recon)
                total_kld = self.kl_divergence(mu, logvar)
                total_loss = recon_loss + self.beta * total_kld

                # Update net params
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def update_decoder(self, decoder_state_dict):
        """Helper method to swap out the current decoder for a new decoder ensuring it is a new object with a deep copy."""

        self.model.decoder.load_state_dict(copy.deepcopy(decoder_state_dict))
