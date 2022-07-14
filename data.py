"""
Read in the data from a specified data source
"""
from torch import randperm
from torch.utils.data import random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


class Data:
    def __init__(self, dataset_name, num_users, sample_ratio = 1, alpha = None, normalize = True, resize = None):
        """Read in the data, split training data into user subsets, and read in server test data.

        :param dataset_name:

        :return:
        """
        dataset_name = dataset_name.lower()

        if dataset_name == 'mnist':
            transform_list = []

            #Establishing transforms
            transform_list.append(ToTensor())
            if resize is not None:
                transform_list.append(Resize(resize))
            if normalize:
                transform_list.append(Normalize((0.1307,), (0.3081,)))

            transform_list = Compose(transform_list)

            dataset_train = MNIST(
                root="data/mnist",
                download=True,
                train=True,
                transform=transform_list,
            )

            # Case where we only distribute a portion of the dataset to users
            if sample_ratio < 1:
                num_samples_keep = int(len(dataset_train) * sample_ratio)
                indices = randperm(len(dataset_train))[ : num_samples_keep] #randomly choosing samples to keep
                dataset_train = Subset(dataset_train, indices)

            num_train_samples = len(dataset_train)
            num_user_samples = int(num_train_samples / num_users)

            data_split_sequence = []
            for u in range(num_users):
                data_split_sequence.append(num_user_samples)

            # Data is composed of dataset.Subset objects
            if alpha is None:
                self.train_data = random_split(dataset_train, data_split_sequence)
            else:
                self.train_data = self.split_data_dirichlet(dataset_train)

            # We ALWAYS keep the full test set for final model evaluation
            dataset_test = MNIST(
                root="data/mnist",
                download=True,
                train=False,
                transform=transform_list,
            )
            self.test_data = dataset_test

            # MNIST is black and white, so number of channels is 1
            # Number of classes is 10 because there are 10 digits
            self.num_channels = 1
            self.num_classes = 10
        else:
            raise NotImplementedError("Only mnist has been implemented. Please implement other datasets.")

    def split_data_dirichlet(self):
        pass

if __name__ == '__main__':
    MNIST_data = Data('MNIST', num_users = 10, sample_ratio = 0.25, normalize = True, resize = 32)
    print(MNIST_data.train_data)
