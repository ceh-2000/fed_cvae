import copy

import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset


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


def reconstruction_loss(num_channels, x, x_recon):
    """Compute the reconstruction loss comparing input and reconstruction
    using an appropriate distribution given the number of channels.

    :param x: the input image
    :param x_recon: the reconstructed image produced by the decoder

    :return: reconstruction loss
    """

    batch_size = x.size(0)
    assert batch_size != 0

    # Use w/one-channel images
    if num_channels == 1:
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction="sum"
        ).div(batch_size)
    # Multi-channel images
    elif num_channels == 3:
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum").div(batch_size)
    else:
        raise NotImplementedError("We only support 1 and 3 channel images.")

    return recon_loss


def kl_divergence(mu, logvar):
    """Compute KL Divergence between the multivariate normal distribution of z
    and a multivariate standard normal distribution.

    :param mu: the mean of the predicted distribution
    :param logvar: the log-variance of the predicted distribution

    :return: total KL divergence loss
    """

    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # Shortcut: KL divergence w/N(0, I) prior and encoder dist is a multivariate normal
    # Push from multivariate normal --> multivariate STANDARD normal X ~ N(0,I)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld


class WrapperDataset(Dataset):
    """Wrapper dataset to put into a dataloader."""

    def __init__(self, X, y, z):
        self.X = X
        self.y = y
        self.z = z

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.z != None:
            return self.X[idx], self.y[idx], self.z[idx]
        else:
            return self.X[idx], self.y[idx]


if __name__ == "__main__":
    x_recon = torch.randn(100, 1, 32, 32)
    x_true = torch.sigmoid(torch.randn(100, 1, 32, 32))

    print(reconstruction_loss(1, x_true, x_recon))
