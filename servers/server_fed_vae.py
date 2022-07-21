import copy

import numpy as np
from torch.utils.data import DataLoader

from servers.server import Server
from users.user_fed_vae import UserFedVAE
from utils import average_weights, one_hot_encode
import torch
import random


class ServerFedVAE(Server):
    def __init__(self, base_params, z_dim, image_size, beta):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size
        self.beta = beta
        self.decoder = None

        self.num_train_samples = 1000

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

    def generate_data_from_aggregated_decoder(self):
        # Sample z's + y's from uniform distribution
        z_sample = self.users[0].model.sample_z(self.num_train_samples, "uniform")
        print(z_sample.shape)

        classes = np.arange(self.num_classes)
        classes_hot = one_hot_encode(classes, self.num_classes)
        y_hot_sample = random.choices(classes_hot, k=self.num_train_samples)

        with torch.no_grad():
            X_sample = self.decoder(z_sample, y_hot_sample)


        # Put images and labels in wrapper pytoch dataset (e.g. override _get_item())

        # Put dataset into pytorch dataloader and return dataloader

        pass

    def train_classifier(self):
        # Optionally reinitialize the classifier weights every global epoch

        # Sample z's + y's and pass them through the aggregated decoder to generate samples

        # Train classifier on these images as normal

        pass

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

            self.classifier_dataloader = self.generate_data_from_aggregated_decoder()

            self.train_classifier()

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")

if __name__ == "__main__":
    y = torch.arange(10)

    y_hot = one_hot_encode(y, 10)

    y_hot_sample = torch.random.choice(y_hot, k = 10)

    final_y = torch.Tensor(10, 10)
    torch.cat(y_hot_sample, out=final_y)

    print(final_y)