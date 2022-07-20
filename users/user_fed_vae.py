from models.VAE import CVAE
from users.user import User
from utils import one_hot_encode


class UserFedVAE(User):
    def __init__(self, base_params, z_dim, image_size):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.model = CVAE(num_classes=self.num_classes, num_channels=self.num_channels, z_dim=z_dim, image_size=image_size)

    def train(self, local_epochs):
        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                self.model.train()

                y_hot = one_hot_encode(y_batch, self.num_classes)

                x_recon, mu, logvar = self.model(X_batch, y_hot)
                print(x_recon.shape, mu.shape, logvar.shape)



                # # Forward pass through model
                # output = self.model(X_batch)
                #
                # # Compute loss with pre-defined loss function
                # loss = self.loss_func(output, y_batch)
                #
                # # Gradient descent
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
