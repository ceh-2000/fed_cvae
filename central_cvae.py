import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models.VAE import CVAE
from models.classifier import Classifier
from utils import kl_divergence, one_hot_encode, reconstruction_loss, WrapperDataset
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class CentralCVAE:
    def __init__(self, params):
        self.device = params["device"]
        self.epochs = params["glob_epoch"]

        self.train_data = DataLoader(params["train_data"], shuffle=True, batch_size=32)
        self.test_data = DataLoader(params["test_data"], shuffle=True, batch_size=32)

        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]

        self.beta = params["beta"]
        self.z_dim = params["z_dim"]
        self.num_samples_per_class = 5  # Just for display purposes

        self.model = CVAE(
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            z_dim=params["z_dim"],
            image_size=params["image_size"],
            version=0,
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=params["local_LR"])

        self.classifier = Classifier(self.num_channels, self.num_classes).to(
            self.device
        )
        self.classifier_loss_func = CrossEntropyLoss()
        self.classifier_optimizer = Adam(self.classifier.parameters(), lr=0.001)
        self.classifier_num_train_samples = params["classifier_num_train_samples"]
        self.classifier_epochs = params["classifier_epochs"]

        self.good_samples = params["good_samples_bool"]
        self.good_sample_range = (-1, 1)
        self.poor_sample_range = (5, 20)
        self.writer = params["writer"]

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
        z_sample = self.model.sample_z(
            self.classifier_num_train_samples, "uniform", uniform_width=self.good_sample_range if self.good_samples else self.poor_sample_range
        )
        y_sample, y_hot_sample = self.sample_y()

        self.model.decoder.eval()

        with torch.no_grad():
            z_sample, y_hot_sample = z_sample.to(self.device), y_hot_sample.to(
                self.device
            )

            X_sample = self.model.decoder(z_sample, y_hot_sample).detach().cpu()

        # Only apply sigmoid for mnist and fashion
        X_sample = torch.sigmoid(X_sample)

        # Put images and labels in wrapper pytoch dataset (e.g. override _get_item())
        dataset = WrapperDataset(X_sample, y_sample, z_sample)

        # Put dataset into pytorch dataloader and return dataloader
        dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

        return dataloader


    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(epoch)
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_data):
                y_hot = one_hot_encode(y_batch, self.num_classes)

                X_batch, y_hot = X_batch.to(self.device), y_hot.to(self.device)

                X_recon, mu, logvar = self.model(X_batch, y_hot, self.device)

                # Calculate losses
                recon_loss = reconstruction_loss(self.num_channels, X_batch, X_recon)
                total_kld = kl_divergence(mu, logvar)
                total_loss = recon_loss + self.beta * total_kld

                # Update net params
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            self.qualitative_check(
                epoch, self.model.decoder, f"Novel images ({'good samples' if self.good_samples else 'poor samples'})"
            )

        self.train_classifier()

    def train_classifier(self):
        # Get the training dataset
        classifier_dataloader = self.generate_data_from_aggregated_decoder()

        # Turn on the train setting for our classifier
        self.classifier.train()

        for epoch in range(self.classifier_epochs):
            for batch_idx, (X_batch, y_batch, _) in enumerate(
                classifier_dataloader
            ):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                output = self.classifier(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.classifier_loss_func(output, y_batch)

                # Gradient descent
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()


    def test(self):
        """
        Evaluate the global server model by comparing the predicted global labels to the actual test labels
        """
        with torch.no_grad():
            self.classifier.eval()

            total_correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.test_data):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                test_logits = self.classifier(X_batch).cpu()
                pred_probs = F.softmax(input=test_logits, dim=1)
                y_pred = torch.argmax(pred_probs, dim=1)
                y_batch = y_batch.cpu()
                total_correct += np.sum((y_pred == y_batch).numpy())

            accuracy = round(total_correct / len(self.test_data.dataset) * 100, 2)
            print(f"Model accuracy was: {accuracy}%")

            if self.writer:
                self.writer.add_scalar("Global Accuracy/test", accuracy, self.epochs)


    def qualitative_check(self, e, dec, name):
        """
        Display images generated by the server side decoder

        :param e: Global epoch number
        :param dec: Decoder to forward pass z + y_hot through
        :param name: Identifier for saving
        """

        num_samples = self.num_classes * self.num_samples_per_class
        z_sample = self.model.sample_z(num_samples, "uniform", uniform_width=self.good_sample_range if self.good_samples else self.poor_sample_range)
        y = torch.tensor(
            [i for i in range(self.num_classes)] * self.num_samples_per_class
        )
        y_hot = one_hot_encode(y, self.num_classes)

        with torch.no_grad():
            z_sample, y_hot = z_sample.to(self.device), y_hot.to(self.device)
            novel_imgs = dec(z_sample, y_hot).cpu()

        self.save_images(novel_imgs, True, name, e)


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
