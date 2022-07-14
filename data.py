"""
Read in the data from a specified data source
"""
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


class Data:
    def __init__(self, dataset_name, num_users):
        if dataset_name == "mnist":
            dataset_train = MNIST(
                root="data/mnist",
                download=True,
                train=True,
                transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            )

            num_train_samples = len(dataset_train)
            num_user_samples = int(num_train_samples / num_users)

            data_split_sequence = []
            for u in range(num_users):
                data_split_sequence.append(num_user_samples)

            # Data is composed of dataset.Subset objects
            self.train_data = random_split(dataset_train, data_split_sequence)

            dataset_test = MNIST(
                root="data/mnist",
                download=True,
                train=False,
                transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            )
            self.test_data = dataset_test

            # MNIST is black and white, so number of channels is 1
            # Number of classes is 10 because there are 10 digits
            self.num_channels = 1
            self.num_classes = 10
        else:
            raise NotImplementedError(
                "Only mnist has been implemented. Please implement other datasets."
            )
