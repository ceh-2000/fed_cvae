"""
Read in the data from a specified data source
"""

import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST


class Data:
    def __init__(self, dataset_name, num_users):
        # if dataset_name == 'mnist':
        dataset_train = MNIST(root="data/mnist", download=True, train=True)

        num_train_samples = len(dataset_train)
        num_user_samples = int(num_train_samples / num_users)

        data_split_sequence = []
        for u in range(num_users):
            data_split_sequence.append(num_user_samples)

        # Data is composed of dataset.Subset objects
        self.train_data = random_split(dataset_train, data_split_sequence)

        dataset_test = MNIST(root="data/mnist", download=True, train=False)
        self.test_data = dataset_test


if __name__ == "__main__":
    dataset = MNIST(root="data/mnist", download=True)

    data = random_split(
        dataset, [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 10000]
    )
    user_0 = data[0]

    X = torch.index_select(user_0.dataset.data, 0, torch.IntTensor(user_0.indices))
    y = torch.index_select(user_0.dataset.targets, 0, torch.IntTensor(user_0.indices))

    print(X.shape)
    print(y.shape)

    # t = random_split(dataset, [int(len(dataset)/10), 10])
