"""
Read in the data from a specified data source
"""
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.utils
from PIL import Image
from torch.utils.data import Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Setting seeds for reproducibility
np.random.seed(1693)


class Data:
    def __init__(
        self,
        dataset_name,
        num_users,
        writer,
        central,
        sample_ratio=1.0,
        alpha=None,
        normalize=False,
        visualize=False,
    ):
        """Read in the data, split training data into user subsets, and read in server test data.

        :param dataset_name: Name of the dataset of to use
        :param num_users: Number of users for distributed training
        :param writer: Logger
        :param central: Boolean for whether we want centralized or distributed training
        :param sample_ratio: How much of the training data to use
        :param alpha: Parameter used to control heterogeneity
        :param normalize: Whether images should normalized in the transform
        :param visualize: Boolean to visualize data distribution
        """

        self.dataset_name = dataset_name.lower()
        self.alpha = alpha
        self.sample_ratio = sample_ratio
        self.num_users = num_users
        self.writer = writer

        if self.dataset_name == "mnist":
            # MNIST is black and white, so number of channels is 1
            # Number of classes is 10 because there are 10 digits
            self.num_channels = 1
            self.num_classes = 10
            self.image_size = 32

            transform_list = []

            # Establishing transforms
            transform_list.append(ToTensor())
            transform_list.append(Resize(32))  # Everyone gets resized to 32
            if normalize:
                transform_list.append(Normalize((0.1307,), (0.3081,)))

            transform_list = Compose(transform_list)

            dataset_train = MNIST(
                root="data/mnist",
                download=True,
                train=True,
                transform=transform_list,
            )

            # We ALWAYS keep the full test set for final model evaluation
            dataset_test = MNIST(
                root="data/mnist",
                download=True,
                train=False,
                transform=transform_list,
            )
        elif self.dataset_name == "fashion":
            self.num_channels = 1
            self.num_classes = 10
            self.image_size = 32

            transform_list = []

            # Establishing transforms
            transform_list.append(ToTensor())
            transform_list.append(Resize(32))  # Everyone gets resized to 32
            if normalize:
                transform_list.append(Normalize((0.5,), (0.5,)))

            transform_list = Compose(transform_list)

            dataset_train = FashionMNIST(
                root="data/mnist",
                download=True,
                train=True,
                transform=transform_list,
            )

            dataset_test = FashionMNIST(
                root="data/mnist",
                download=True,
                train=False,
                transform=transform_list,
            )
        elif self.dataset_name == "svhn":
            self.num_channels = 3
            self.num_classes = 10
            self.image_size = 32

            transform_list = []

            # Establishing transforms
            transform_list.append(ToTensor())
            transform_list.append(Resize(32))  # Everyone gets resized to 32
            if normalize:
                transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

            transform_list = Compose(transform_list)

            dataset_train = SVHN(
                root="data/svhn",
                download=True,
                split="train",
                transform=transform_list,
            )

            dataset_test = SVHN(
                root="data/svhn",
                download=True,
                split="test",
                transform=transform_list,
            )
        elif self.dataset_name == "cifar10":
            self.num_channels = 3
            self.num_classes = 10
            self.image_size = 32

            transform_list = []

            # Establishing transforms
            transform_list.append(ToTensor())
            transform_list.append(Resize(32))  # Everyone gets resized to 32
            if normalize:
                transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

            transform_list = Compose(transform_list)

            dataset_train = CIFAR10(
                root="data/cifar10",
                download=True,
                train=True,
                transform=transform_list,
            )

            dataset_test = CIFAR10(
                root="data/cifar10",
                download=True,
                train=False,
                transform=transform_list,
            )
        else:
            raise NotImplementedError(
                f"Dataset '{dataset_name}' has not been implemented, please choose either mnist, fashion, svhn, or cifar10"
            )

        # Case where we only distribute a portion of the dataset to users
        if self.sample_ratio < 1:
            num_samples_keep = int(len(dataset_train) * self.sample_ratio)
            indices = np.random.permutation(len(dataset_train))[
                :num_samples_keep
            ]  # randomly choosing samples to keep
            dataset_train = Subset(dataset_train, indices)

        num_train_samples = len(dataset_train)
        num_user_samples = int(num_train_samples / self.num_users)

        data_split_sequence = []
        for u in range(self.num_users):
            data_split_sequence.append(num_user_samples)
        if sum(data_split_sequence) != num_train_samples:
            data_split_sequence[0] += num_train_samples - sum(data_split_sequence)

        # Data is composed of dataset.Subset objects
        if central:
            self.train_data = dataset_train

        # Only partition into user datasets if we don't want centralized learning
        else:
            if self.alpha is None:
                self.train_data = random_split(
                    dataset_train,
                    data_split_sequence,
                    generator=torch.Generator().manual_seed(1693),
                )
            else:
                self.train_data = self.split_data_dirichlet(dataset_train, visualize)

        self.test_data = dataset_test

    def split_data_dirichlet(self, dataset_train, visualize):
        """Split the dataset according to proportionsk sampled from a Dirichlet distribution, with alpha controlling the level of heterogeneity.

        :param dataset_train: the training dataset to split across users
        :param visualize: whether or not to visualize dataset split across users

        :return: user data splits as a list of torch.dataset.Subset objects
        """

        # Getting indices for each class
        if self.sample_ratio == 1:
            targets = dataset_train.targets.numpy()
        else:
            try:
                targets = np.array(dataset_train.dataset.targets)[
                    dataset_train.indices
                ]  # case where we've subsetted the dataset, in which case we reframe indices based on this subset's data indices
            except AttributeError:
                targets = np.array(
                    [
                        int(dataset_train.dataset[i][1])
                        for i in range(len(dataset_train))
                    ]
                )  # case where we have to manually extract targets

        class_idxs = {}

        for i in range(self.num_classes):
            class_i_idxs = np.nonzero(targets == i)[0]
            np.random.shuffle(class_i_idxs)  # shuffling so that order doesn't matter
            class_idxs[i] = class_i_idxs

        # Sampling proportions for each user based on a Dirichlet distribution
        user_props = []  # will end up shape [num_users x num_classes]
        for i in range(self.num_users):
            props = np.random.dirichlet(
                np.repeat(self.alpha, self.num_classes)
            )  # sample the proportion of total samples that each class represents for a given user (will add to 1)
            user_props.append(props)

        user_props = np.array(user_props)
        scaled_props = user_props / np.sum(
            user_props, axis=0
        )  # scaling so that we add up to 100% of the data for each class (i.e., now we can distribute via these proportions and end up giving out all of the data)

        # Distributing data to users
        user_data_idxs = {i: [] for i in range(self.num_users)}
        num_samples_per_user_per_class = {
            c: None for c in range(self.num_classes)
        }  # for visualization purposes

        for c in range(self.num_classes):
            num_pts_per_user = (scaled_props[:, c] * len(class_idxs[c])).astype(
                int
            )  # giving each user a number of samples based on their sampled proportion
            num_samples_per_user_per_class[c] = num_pts_per_user
            indices_per_user = [
                np.sum(num_pts_per_user[0 : i + 1]) for i in range(self.num_users)
            ]  # sorting out indices for pulling out this data

            for i in range(self.num_users):
                start_idx = indices_per_user[i - 1] if i - 1 >= 0 else 0
                end_idx = indices_per_user[i]
                user_data_idxs[i].extend(list(class_idxs[c][start_idx:end_idx]))

            # If we didn't quite distribute all data, distribute final samples uniformly at random
            if (indices_per_user[-1] - 1) < len(class_idxs[c]):
                remaining_idxs = class_idxs[c][indices_per_user[-1] :]

                for idx in remaining_idxs:
                    random_user = int(
                        np.random.choice([i for i in range(self.num_users)], size=1)
                    )  # choose a user
                    user_data_idxs[random_user].append(idx)  # give the user this sample

                    num_samples_per_user_per_class[c][random_user] += 1

        # Visualize the resulting dataset split to confirm level of heterogeneity
        if visualize:
            self._visualize_heterogeneity(num_samples_per_user_per_class)

        # Wrapping dataset subset into PyTorch Subset objects
        user_data = []
        for i in range(self.num_users):
            user_data.append(Subset(dataset_train, user_data_idxs[i]))

        return user_data

    def _visualize_heterogeneity(self, num_samples_per_user_per_class):
        df = pd.DataFrame(num_samples_per_user_per_class)
        df = pd.melt(
            df,
            value_vars=[i for i in range(self.num_classes)],
            var_name="class",
            value_name="num_samples",
        )
        df["user"] = [i for i in range(self.num_users)] * self.num_classes

        size_max = df["num_samples"].max() / (10 * self.sample_ratio)
        sns.set_style("whitegrid")
        sns.scatterplot(
            x="user",
            y="class",
            size="num_samples",
            data=df,
            legend=False,
            size_norm=(0, size_max),
            sizes=(0, size_max),
        )

        plt.xticks([i for i in range(self.num_users)])
        plt.yticks([c for c in range(self.num_classes)])
        plt.xlabel("User ID", fontweight="bold")
        plt.ylabel("Class label", fontweight="bold")

        if self.writer:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")

            im = Image.open(img_buf)

            # Save the image to tensorboard
            im_tensor = ToTensor()(im).unsqueeze(0)
            grid = torchvision.utils.make_grid(im_tensor)
            self.writer.add_image("Data distribution", grid, 0)

        plt.show()


if __name__ == "__main__":
    MNIST_data = Data(
        "mnist",
        num_users=20,
        writer=None,
        sample_ratio=0.5,
        alpha=0.01,
        normalize=False,
        visualize=True,
        central=False,
    )
    print([len(MNIST_data.train_data[i]) for i in range(MNIST_data.num_users)])
    print(sum([len(MNIST_data.train_data[i]) for i in range(MNIST_data.num_users)]))

    from collections import Counter

    from torch.utils.data import DataLoader

    dl = DataLoader(MNIST_data.train_data[5], batch_size=len(MNIST_data.train_data[5]))
    print(Counter(next(iter(dl))[1].tolist()))
