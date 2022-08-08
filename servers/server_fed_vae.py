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
from utils import (WrapperClassifierDataset, WrapperDecoderDataset,
                   average_weights, one_hot_encode, reconstruction_loss)


class ServerFedVAE(Server):
    def __init__(
        self, base_params, z_dim, image_size, beta, classifier_num_train_samples, classifier_epochs, decoder_num_train_samples, decoder_epochs
    ):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size
        self.beta = beta

        self.decoder = ConditionalDecoder(
            self.image_size, self.num_classes, self.num_channels, self.z_dim
        )
        self.kd_optimizer = Adam(self.decoder.parameters(), lr=0.001)

        self.decoder_num_train_samples = decoder_num_train_samples
        self.decoder_epochs = decoder_epochs

        self.classifier_loss_func = CrossEntropyLoss()
        self.classifier_optimizer = Adam(self.server_model.parameters(), lr=0.001)

        self.classifier_num_train_samples = classifier_num_train_samples
        self.classifier_epochs = classifier_epochs
        self.num_samples_per_class = 5

    def compute_pmf(self, u):
        """
        Helper function to get probabilities for user target values

        :param u: User ID
        :return: Probability distribution of target data for a given user
        """
        subset_1_indices = self.data_subsets[u].dataset.indices
        subset_1_dataset = self.data_subsets[u].dataset.dataset.targets
        subset_1 = subset_1_dataset[subset_1_indices]

        subset_2_indices = self.data_subsets[u].indices
        subset_2_dataset = subset_1
        subset_2 = subset_2_dataset[subset_2_indices]

        vals, counts = torch.unique(subset_2, return_counts=True)
        count_dict = {}
        for i in range(len(vals)):
            count_dict[int(vals[i])] = int(counts[i])

        pmf = np.zeros(self.num_classes)

        for p in range(self.num_classes):
            if p in count_dict:
                pmf[p] = count_dict.get(p) / torch.sum(counts)

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
                pmf,
            )
            self.users.append(new_user)

    def average_decoders(self, decoders):
        """
        Helper method to average the decoders into a shared model

        :param decoders: List of decoders from selected user models
        :param distill: Boolean that controls whether we aggregate with KD or not

        :return: aggregated decoder
        """
        return average_weights(decoders)

    def distill_user_decoders(self, users, e):
        """
        Aggregate user decoders using knowledge distillation

        :param users: List of selected users to use as teacher models
        """

        # Number of samples per user
        len_data = int(
            self.decoder_num_train_samples / len(users)
        )

        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        z_vals = torch.Tensor()

        for u in users:
            z = u.model.sample_z(len_data, "uniform")

            # Sample y's according to each user's target distribution
            classes = np.arange(self.num_classes)
            y = torch.from_numpy(np.random.choice(classes, size=len_data, p=u.pmf))
            y_hot = one_hot_encode(y, self.num_classes)

            # Detaching ensures that there aren't issues w/trying to calculate the KD grad WRT this net's params - not needed!
            X = u.model.decoder(z, y_hot).detach()

            X_vals = torch.cat((X_vals, X), 0)
            y_vals = torch.cat((y_vals, y_hot), 0)
            z_vals = torch.cat((z_vals, z), 0)

        # Normalize "target" images to ensure reconstruction loss works correctly
        X_vals = torch.sigmoid(X_vals)

        decoder_dataset = WrapperDecoderDataset(X_vals, y_vals, z_vals)
        dl = DataLoader(decoder_dataset, shuffle=True, batch_size=32)

        self.decoder.train()

        for epoch in range(
            self.decoder_epochs
        ):  
            for X_batch, y_batch, z_batch in dl:
                X_server = self.decoder(z_batch, y_batch)

                recon_loss = reconstruction_loss(self.num_channels, X_batch, X_server)

                self.kd_optimizer.zero_grad()
                recon_loss.backward(retain_graph=False)
                self.kd_optimizer.step()

        return self.decoder.state_dict()

    def sample_y(self):
        """
        Helper method to sample one-hot-encoded targets

        :return: one-hot-encoded y's
        """
        y = np.arange(self.num_classes)
        y_sample = np.random.choice(y, size=self.classifier_num_train_samples)

        y_s = torch.from_numpy(y_sample)

        y_hot = one_hot_encode(y_s, self.num_classes)

        return y_s, y_hot

    def generate_data_from_aggregated_decoder(self):
        # Sample z's + y's from uniform distribution
        z_sample = self.users[0].model.sample_z(self.classifier_num_train_samples, "uniform")
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

            # Update the server decoder using weight averaging and knowledge distillation
            avg_state_dict = self.average_decoders(decoders)
            self.decoder.load_state_dict(copy.deepcopy(avg_state_dict))
            decoder_state_dict = copy.deepcopy(
                self.distill_user_decoders(selected_users, e)
            )

            # Qualitative image check - both the server and a misc user!
            self.qualitative_check(e, self.decoder, "Novel images server decoder")
            self.qualitative_check(
                e, self.users[0].model.decoder, "Novel images user 0 decoder"
            )

            # Send aggregated decoder to selected users
            for u in selected_users:
                u.update_decoder(decoder_state_dict)

            print(f"Aggregated decoders for epoch {e}")

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

        if glob_iter is not None and self.writer:
            print("saving images")
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
