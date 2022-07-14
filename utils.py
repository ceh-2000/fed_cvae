import copy

import torch


def avg_weights(w):
    """Helper method to returns the average of a list of weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg


def average_weights(model_list):
    """
    Average all of the weights for a list of given models

    :param model_list: List of models to average
    :return: An averaged state dictionary
    """

    # Extract state dictionaries for all models in model_list
    weight_objects = []
    for w in model_list:
        weight_objects.append(copy.deepcopy(w.state_dict()))

    # Average the weights from models
    avg_model_state_dict = avg_weights(weight_objects)

    return avg_model_state_dict
