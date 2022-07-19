from torch import nn


class View(nn.Module):
    """Class to change the size of a PyTorch tensor."""

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
