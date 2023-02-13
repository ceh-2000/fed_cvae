import copy
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation
from torchvision.utils import make_grid

from models.decoder import ConditionalDecoder
from servers.server import Server
from users.user_fed_cvae_kd import UserFedVAE
from utils import (WrapperDataset, average_weights, one_hot_encode,
                   reconstruction_loss)


class ServerFedVAE(Server):
    def __init__(
        self,
        base_params,
        z_dim,
        image_size,
        beta,
        classifier_num_train_samples,
        classifier_epochs,
        decoder_num_train_samples,
        decoder_epochs,
        decoder_LR,
        uniform_range,
        should_weight,
        should_initialize_same,
        should_avg,
        should_fine_tune,
        heterogeneous_models,
        should_transform,
    ):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size
        self.beta = beta

        self.decoder = ConditionalDecoder(
            self.image_size, self.num_classes, self.num_channels, self.z_dim
        ).to(self.device)
        self.kd_optimizer = Adam(self.decoder.parameters(), lr=decoder_LR)

        self.decoder_num_train_samples = decoder_num_train_samples
        self.decoder_epochs = decoder_epochs

        self.classifier_loss_func = CrossEntropyLoss()
        self.classifier_optimizer = Adam(self.server_model.parameters(), lr=0.001)
        self.initial_classifier_state_dict = copy.deepcopy(
            self.server_model.state_dict()
        )

        self.classifier_num_train_samples = classifier_num_train_samples
        self.classifier_epochs = classifier_epochs

        self.num_samples_per_class = 5  # Just for display purposes
        self.uniform_range = uniform_range

        # Variables important for ablation experiments
        self.should_weight = should_weight
        self.should_initialize_same = should_initialize_same
        self.should_avg = should_avg
        self.should_fine_tune = should_fine_tune
        self.heterogeneous_models = heterogeneous_models
        self.should_transform = should_transform

    def compute_data_amt_and_pmf(self, u):
        """
        Helper function to get probabilities for user target values

        :param u: User ID
        :return: Probability distribution of target data for a given user
        """

        targets = torch.from_numpy(
            np.array(
                [
                    int(self.data_subsets[u][i][1])
                    for i in range(len(self.data_subsets[u]))
                ]
            )
        )

        vals, counts = torch.unique(targets, return_counts=True)
        count_dict = {}
        for i in range(len(vals)):
            count_dict[int(vals[i])] = int(counts[i])

        pmf = np.zeros(self.num_classes)

        for p in range(self.num_classes):
            if p in count_dict:
                pmf[p] = count_dict.get(p) / torch.sum(counts)

        # Handling the case where Python's precision messes with our calulations
        # This will be a small difference, so we can safely add/subract it from wherever!
        if np.sum(pmf) != 1.0:
            diff = np.sum(pmf) - 1.0

            if diff > 0:
                pmf[np.argmax(pmf)] -= abs(diff)
            elif diff < 0:
                pmf[np.argmax(pmf)] += abs(diff)

        assert np.sum(pmf) == 1.0, f"Vector for user ID {u} sums to {np.sum(pmf)}"

        return targets.shape[0], pmf

    def create_users(self):
        data_amts = np.zeros(
            [
                self.num_users,
            ]
        )
        pmfs = []
        for u in range(self.num_users):
            data_amt, pmf = self.compute_data_amt_and_pmf(u)
            data_amts[u] = data_amt
            pmfs.append(pmf)

        total_data = np.sum(data_amts)
        data_amts = data_amts / total_data

        version_options = [int(i) for i in list(self.heterogeneous_models)]
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)

            version = random.choice(version_options)

            new_user = UserFedVAE(
                {
                    "device": self.device,
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                    "local_LR": self.local_LR,
                    "use_adam": self.use_adam,
                },
                self.z_dim,
                self.image_size,
                self.beta,
                data_amts[u],
                pmfs[u],
                version,
            )
            self.users.append(new_user)

    def average_decoders(self, decoders, data_amts):
        """
        Helper method to average the decoders into a shared model

        :param decoders: List of decoders from selected user models
        :param data_amts: List of how many samples each user has

        :return: aggregated decoder
        """
        if self.should_weight:
            return average_weights(decoders, data_amts=data_amts)
        else:
            return average_weights(decoders)

    def generate_dataset_from_user_decoders(self, users, num_train_samples):
        """
        Generate a novel dataset from user decoders

        :param users: List of selected users to use as teacher models
        :param num_train_samples: How many samples to add to our new dataset
        """

        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        z_vals = torch.Tensor()

        total_train_samples = 0
        count_user = 0
        for u in users:
            u.model.eval()

            # Sample a proportional number of samples to the amount of data the current user has seen
            if self.should_weight:
                user_num_train_samples = int(u.data_amt * num_train_samples)
            else:
                user_num_train_samples = int(num_train_samples / self.num_users)

            if count_user == self.num_users - 1:
                user_num_train_samples = num_train_samples - total_train_samples
            else:
                total_train_samples += user_num_train_samples
                count_user += 1

            z = u.model.sample_z(
                user_num_train_samples, "truncnorm", width=self.uniform_range
            ).to(self.device)

            # Sample y's according to each user's target distribution
            classes = np.arange(self.num_classes)
            y = torch.from_numpy(
                np.random.choice(classes, size=user_num_train_samples, p=u.pmf)
            )
            y_hot = one_hot_encode(y, self.num_classes).to(self.device)

            # Detaching ensures that there aren't issues w/trying to calculate the KD grad WRT this net's params - not needed!
            X = u.model.decoder(z, y_hot).detach()

            X, y_hot, z = X.cpu(), y_hot.cpu(), z.cpu()

            X_vals = torch.cat((X_vals, X), 0)
            y_vals = torch.cat((y_vals, y_hot), 0)
            z_vals = torch.cat((z_vals, z), 0)

        # Normalize "target" images to ensure reconstruction loss works correctly
        X_vals = torch.sigmoid(X_vals)

        decoder_dataset = WrapperDataset(X_vals, y_vals, z_vals)
        dl = DataLoader(decoder_dataset, shuffle=True, batch_size=32)

        return dl

    def distill_user_decoders(self, users):
        """
        Aggregate user decoders using knowledge distillation

        :param users: List of selected users to use as teacher models
        """
        dl = self.generate_dataset_from_user_decoders(
            users, self.decoder_num_train_samples
        )

        self.decoder.train()

        for epoch in range(self.decoder_epochs):
            for X_batch, y_batch, z_batch in dl:
                X_batch, y_batch, z_batch = (
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                    z_batch.to(self.device),
                )

                X_server = self.decoder(z_batch, y_batch)

                recon_loss = reconstruction_loss(self.num_channels, X_batch, X_server)

                self.kd_optimizer.zero_grad()
                recon_loss.backward()
                self.kd_optimizer.step()

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
        z_sample = self.users[0].model.sample_z(
            self.classifier_num_train_samples,
            "truncnorm",
            width=self.uniform_range,
        )
        y_sample, y_hot_sample = self.sample_y()

        self.decoder.eval()

        with torch.no_grad():
            z_sample, y_hot_sample = z_sample.to(self.device), y_hot_sample.to(
                self.device
            )

            X_sample = self.decoder(z_sample, y_hot_sample).detach().cpu()

        # Apply transforms to inject variations into samples for SVHN
        if self.dataset_name == "svhn" and self.should_transform:
            transforms = Compose(
                [
                    RandomRotation(45),
                ]
            )

            X_sample_transform = transforms(X_sample)

            self.save_images(
                X_sample_transform[:10], True, "transformed_fedvae_images", 1
            )

            X_sample = torch.cat((X_sample, X_sample_transform), 0)
            y_sample = torch.cat((y_sample, y_sample), 0)
            z_sample = torch.cat((z_sample, z_sample), 0)

        # Only apply sigmoid for mnist and fashion
        X_sample = torch.sigmoid(X_sample)

        # Put images and labels in wrapper pytoch dataset (e.g. override _get_item())
        dataset = WrapperDataset(X_sample, y_sample, z_sample)

        # Put dataset into pytorch dataloader and return dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

        return dataloader

    def train_classifier(self, reinitialize_weights=False):
        if reinitialize_weights:
            self.server_model.load_state_dict(
                copy.deepcopy(self.initial_classifier_state_dict)
            )

        self.server_model.train()

        for epoch in range(self.classifier_epochs):
            for batch_idx, (X_batch, y_batch, _) in enumerate(
                self.classifier_dataloader
            ):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                output = self.server_model(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.classifier_loss_func(output, y_batch)

                # Gradient descent
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

    def train(self):
        # Ensure all models are initialized the same
        if self.should_initialize_same:
            weight_init_state_dict = self.users[0].model.state_dict()
            for u in self.users:
                u.model.load_state_dict(copy.deepcopy(weight_init_state_dict))

        # Training loop
        for e in range(self.glob_epochs):
            self.evaluate(e)

            selected_users = self.sample_users()

            # Send server decoder to selected users
            if not self.heterogeneous_models and not self.should_initialize_same:
                decoder_state_dict = copy.deepcopy(self.decoder.state_dict())
                for u in selected_users:
                    u.update_decoder(decoder_state_dict)

            # Train selected users and collect their decoder weights and number of training samples
            decoders = []
            data_amts = []
            for u in selected_users:
                u.train(self.local_epochs)
                decoders.append(copy.deepcopy(u.model.decoder).cpu())
                data_amts.append(u.data_amt)

            print(f"Finished training user models for epoch {e}")

            # Update the server decoder using weight averaging and knowledge distillation
            if self.should_avg:
                avg_state_dict = self.average_decoders(decoders, data_amts)
                self.decoder.load_state_dict(copy.deepcopy(avg_state_dict))

            if self.should_fine_tune:
                self.distill_user_decoders(selected_users)

            # Qualitative image check - both the server and a misc user!
            self.qualitative_check(e, self.decoder, "Novel images server decoder")
            self.qualitative_check(
                e, self.users[0].model.decoder, "Novel images user 0 decoder"
            )
            self.qualitative_check(
                e, self.users[1].model.decoder, "Novel images user 1 decoder"
            )
            self.qualitative_check(
                e, self.users[2].model.decoder, "Novel images user 2 decoder"
            )
            self.qualitative_check(
                e, self.users[3].model.decoder, "Novel images user 3 decoder"
            )
            self.qualitative_check(
                e, self.users[4].model.decoder, "Novel images user 4 decoder"
            )

            # Generate a dataloader holding the generated images and labels
            self.classifier_dataloader = self.generate_data_from_aggregated_decoder()
            print(
                f"Generated {len(self.classifier_dataloader.dataset)} samples to train server classifier for epoch {e}"
            )

            # Train the server model's classifier
            self.train_classifier(reinitialize_weights=True)
            print(f"Trained server classifier for epoch {e}")

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
        z_sample = self.users[0].model.sample_z(
            num_samples, "truncnorm", width=self.uniform_range
        )
        y = torch.tensor(
            [i for i in range(self.num_classes)] * self.num_samples_per_class
        )
        y_hot = one_hot_encode(y, self.num_classes)

        with torch.no_grad():
            z_sample, y_hot = z_sample.to(self.device), y_hot.to(self.device)
            novel_imgs = dec(z_sample, y_hot).cpu()

        self.save_images(novel_imgs, True, name, e)


if __name__ == "__main__":
    num_classes = 10
    y = np.arange(num_classes)
    y_sample = np.random.choice(y, size=100)
    print(y_sample.shape)

    y_s = torch.from_numpy(y_sample)
    y_hot = one_hot_encode(y_s, num_classes)
    print(y_hot.shape)
