import copy

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models.decoder import ConditionalDecoder
from servers.server import Server
from users.user_fed_vae import UserFedVAE
from utils import WrapperClassifierDataset, WrapperDecoderDataset, average_weights, one_hot_encode, reconstruction_loss


class ServerFedVAE(Server):
    def __init__(
        self, base_params, z_dim, image_size, beta, num_train_samples, classifier_epochs
    ):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size
        self.beta = beta

        self.decoder = ConditionalDecoder(
            self.image_size, self.num_classes, self.num_channels, self.z_dim
        )
        self.kd_optimizer = Adam(self.decoder.parameters(), lr=0.001)

        self.classifier_loss_func = CrossEntropyLoss()
        self.classifier_optimizer = Adam(self.server_model.parameters(), lr=0.001)

        self.num_train_samples = num_train_samples
        self.classifier_epochs = classifier_epochs
        self.num_samples_per_class = 5

    def compute_pmf(self, u):
        """
        Helper function to get probabilities for user target values

        :param u: User ID
        :return: Probability distribution of target data for a given user
        """

        all_targets = self.data_subsets[u].dataset.dataset.targets
        indices = self.data_subsets[u].dataset.indices
        targets = all_targets[indices]
        _, counts = torch.unique(targets, return_counts=True)
        pmf = counts / torch.sum(counts)

        return pmf

    def create_users(self):
        for u in range(self.num_users):

            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
            pmf = self.compute_pmf(u)


            new_user = UserFedVAE(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                },
                self.z_dim,
                self.image_size,
                self.beta,
                pmf
            )
            self.users.append(new_user)

    def average_decoders(self, decoders):
        """
        Helper method to aggregate the decoders into a shared model

        :param decoders: List of decoders from selected user models
        :param distill: Boolean that controls whether we aggregate with KD or not

        :return: aggregated decoder
        """
        return average_weights(decoders)

    def aggregate_decoders(self, users):
        """
        Aggregate decoder using knowledge distillation

        :param users: List of selected users to use as teacher models
        """

        # Number of samples per user
        len_data = int(self.num_train_samples / len(users))

        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        z_vals = torch.Tensor()

        for u in users:
            z = u.model.sample_z(len_data, "uniform")

            # Sample y's according to each user's target distribution
            classes = np.arange(self.num_classes)
            y = torch.from_numpy(np.random.choice(classes, size = len_data, p = u.pmf))
            y_hot = one_hot_encode(y, self.num_classes)

            X = u.model.decoder(z, y_hot)

            X_vals = torch.cat((X_vals, X), 0)
            y_vals = torch.cat((y_vals, y_hot), 0)
            z_vals = torch.cat((z_vals, z), 0)

        dataset = WrapperDecoderDataset(X_vals, y_vals, z_vals)
        dl = DataLoader(dataset, shuffle=True, batch_size=32)

        self.decoder.train()
        for epoch in range(3):
            for batch_idx, (X_batch, y_batch, z_batch) in enumerate(dl):
                X_server = self.decoder(z_batch, y_batch)
                recon_loss = reconstruction_loss(self.num_channels, X_batch, X_server)

                self.kd_optimizer.zero_grad()
                recon_loss.backward()
                self.kd_optimizer.step()

        return self.decoder.state_dict()

    def sample_y(self):
        """
        Helper method to sample one-hot-encoded targets

        :return: one-hot-encoded y's
        """
        y = np.arange(self.num_classes)
        y_sample = np.random.choice(y, size=self.num_train_samples)

        y_s = torch.from_numpy(y_sample)
        y_hot = one_hot_encode(y_s, self.num_classes)

        return y_s, y_hot

    def generate_data_from_aggregated_decoder(self):
        # Sample z's + y's from uniform distribution
        z_sample = self.users[0].model.sample_z(self.num_train_samples, "uniform")
        y_sample, y_hot_sample = self.sample_y()

        with torch.no_grad():
            X_sample = self.decoder(z_sample, y_hot_sample)
            X_sample = torch.sigmoid(X_sample)

        # Put images and labels in wrapper pytoch dataset (e.g. override _get_item())
        dataset = WrapperClassifierDataset(X_sample, y_sample)

        # Put dataset into pytorch dataloader and return dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

        return dataloader

    def train_classifier(self):
        self.server_model.train()

        for epoch in range(self.classifier_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.classifier_dataloader):
                # Forward pass through model
                output = self.server_model(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.classifier_loss_func(output, y_batch)

                # Gradient descent
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

    def train(self):
        for e in range(self.glob_epochs):
            self.evaluate(e)

            selected_users = self.sample_users()

            # Train selected users and collect their decoder weights
            decoders = []
            for u in selected_users:
                u.train(self.local_epochs)
                decoders.append(u.model.decoder)

            print(f"Finished training user models for epoch {e}")

            # Update the server decoder
            avg_state_dict = self.average_decoders(decoders)
            self.decoder.load_state_dict(avg_state_dict)

            # Use weight averaging and knowledge distillation to aggregate decoders
            decoder_state_dict = self.aggregate_decoders(selected_users)

            # Send aggregated decoder to selected users
            for u in selected_users:
                u.update_decoder(decoder_state_dict)

            print(f"Aggregated decoders for epoch {e}")

            # Qualitative image check - both the server and a misc user!
            self.qualitative_check(e, self.decoder, "Novel images server decoder")
            self.qualitative_check(
                e, self.users[0].model.decoder, "Novel images user 0 decoder"
            )

            # Generate a dataloader holding the generated images and labels
            self.classifier_dataloader = self.generate_data_from_aggregated_decoder()
            print(
                f"Generated {len(self.classifier_dataloader.dataset)} samples to train server classifier for epoch {e}."
            )

            # Train the server model's classifier
            self.train_classifier()
            print(f"Trained server classifier for epoch {e}.")

            print("__________________________________________")

    def save_images(self, images, sigmoid, name, glob_iter):
        """
        Save images so we can view their outputs

        :param images: List of images to save as a grid
        :param sigmoid: Boolean to tell us whether we are saving a novel image or one from the network
        :param name: Identifier for saving
        :param glob_iter: Global epoch number
        """

        # Use sigmoid if saving image from network
        # e.g. novel image or image from network
        if sigmoid:
            images = torch.sigmoid(images).data

        if glob_iter and self.writer:
            grid = make_grid(images, nrow=self.num_classes)
            self.writer.add_image(name, grid, glob_iter)

    def qualitative_check(self, e, dec, name):
        """
        Display images generated by the server side decoder

        :param e: Global epoch number
        :param dec: Decoder to forward pass z + y_hot through
        :param name: Identifier for saving
        """

        num_samples = self.num_classes * self.num_samples_per_class
        z_sample = self.users[0].model.sample_z(num_samples, "uniform")
        y = torch.tensor(
            [i for i in range(self.num_classes)] * self.num_samples_per_class
        )
        y_hot = one_hot_encode(y, self.num_classes)

        with torch.no_grad():
            novel_imgs = dec(z_sample, y_hot)

        self.save_images(novel_imgs, True, name, e)


if __name__ == "__main__":
    num_classes = 10
    y = np.arange(num_classes)
    y_sample = np.random.choice(y, size=100)
    print(y_sample.shape)

    y_s = torch.from_numpy(y_sample)
    y_hot = one_hot_encode(y_s, num_classes)
    print(y_hot.shape)
