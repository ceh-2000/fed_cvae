import copy

import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

def avg_weights(w, data_amts):
    """Helper method to return the (possibly weighted) average of a list of weights"""

    # If data amounts are given, we take a weighted average... else, we average w/uniform weight
    if data_amts is not None:
        num_data_pts = sum(data_amts)
        weights_for_avging = [amt / num_data_pts for amt in data_amts]
    else:
        weights_for_avging = [1 / len(w) for i in range(len(w))]

    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], weights_for_avging[0])
        for i in range(1, len(w)):
            w_avg[key] += torch.mul(w[i][key], weights_for_avging[i])

    return w_avg


def average_weights(model_list, data_amts=None):
    """
    Take a (weighted) average of all the weights for a given list of models

    :param model_list: List of models to average
    :param data_amts: List of number of local data points for each model

    :return: An averaged state dictionary
    """

    # Extract state dictionaries for all models in model_list
    weight_objects = []
    for w in model_list:
        weight_objects.append(copy.deepcopy(w.state_dict()))

    # Average the weights from models
    avg_model_state_dict = avg_weights(weight_objects, data_amts=data_amts)

    return avg_model_state_dict

def one_hot_encode(y, num_classes):
    """One-hot encode a classification in vector form
    Ex. 10 classes, y = 5 --> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    :param y: Torch tensor of classifications.
    :param num_classes: Integer number of classes in dataset.
    :return: Torch tensor of the one-hot encoded representation of y.
    """

    values = y.detach().numpy()
    onehot_encoder = OneHotEncoder(
        sparse=False, categories=[[i for i in range(0, num_classes)]]
    )
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return torch.tensor(onehot_encoded)


class CustomMnistDataset(Dataset):
    """Custom Mnist Dataset."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        pass

