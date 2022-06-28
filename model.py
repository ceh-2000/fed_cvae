import torch


class MyModel:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)