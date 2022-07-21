import copy

import numpy as np
from torch.utils.data import DataLoader

from servers.server import Server
from users.user_fed_vae import UserFedVAE
from utils import average_weights, one_hot_encode, CustomMnistDataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class ServerFedVAE(Server):
    def __init__(self, base_params, z_dim, image_size, beta):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size
        self.beta = beta
        self.decoder = None

        self.num_train_samples = 1000
        self.classifier_epochs = 10

    def create_users(self):
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)

            new_user = UserFedVAE(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                },
                self.z_dim,
                self.image_size,
                self.beta
            )
            self.users.append(new_user)

    def aggregate_decoders(self, decoders):
        """
        Helper method to aggregate the decoders into a shared model

        :param decoders: List of decoders from selected user models

        :return: aggregated decoder
        """
        return average_weights(decoders)

    def sample_y(self):
        """
        Helper method to sample one-hot-encoded targets

        :return: one-hot-encoded y's
        """
        y = np.arange(self.num_classes)
        y_sample = np.random.choice(y, size=self.num_train_samples)
        print(y_sample.shape)

        y_s = torch.from_numpy(y_sample)
        y_hot = one_hot_encode(y_s, self.num_classes)

        return y_hot

    def generate_data_from_aggregated_decoder(self):
        # Sample z's + y's from uniform distribution
        z_sample = self.users[0].model.sample_z(self.num_train_samples, "uniform")
        print(z_sample.shape)

        y_hot_sample = self.sample_y()
        print(y_hot_sample.shape)

        with torch.no_grad():
            X_sample = self.decoder(z_sample, y_hot_sample)

        # Put images and labels in wrapper pytoch dataset (e.g. override _get_item())
        dataset = CustomMnistDataset(X_sample, y_hot_sample)

        # Put dataset into pytorch dataloader and return dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

        return dataloader

    def train_classifier(self):
        # Optionally reinitialize the classifier weights every global epoch

        # Train classifier on these images as normal
        loss_func = CrossEntropyLoss()
        optimizer = Adam(self.server_model.parameters(), lr=0.001)

        self.server_model.train()

        for epoch in range(self.classifier_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                # Forward pass through model
                output = self.server_model(X_batch)

                # Compute loss with pre-defined loss function
                loss = loss_func(output, y_batch)

                # Gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train(self):
        for e in range(self.glob_epochs):
            self.evaluate(e)

            selected_users = self.sample_users()
            decoders = []
            for u in selected_users:
                u.train(self.local_epochs)
                decoders.append(u.model.decoder)

            decoder_state_dict = self.aggregate_decoders(decoders)
            for u in selected_users:
                u.update_decoder(decoder_state_dict)

            # Update the server decoder
            self.decoder = copy.deepcopy(selected_users[0].model.decoder)

            # Generate a dataloader holding the dataset of generated images and labels
            self.classifier_dataloader = self.generate_data_from_aggregated_decoder()


            self.train_classifier()

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")


if __name__ == "__main__":
    num_classes = 10
    y = np.arange(num_classes)
    y_sample = np.random.choice(y, size = 100)
    print(y_sample.shape)

    y_s = torch.from_numpy(y_sample)
    y_hot = one_hot_encode(y_s, num_classes)
    print(y_hot.shape)