"""
Read in the data from a specified data source
"""

import torch


class Data:
    def __init__(self):
        self.X = torch.arange(-5, 5, 0.1).view(-1, 1)
        self.y = -5 * self.X + 0.1 * torch.randn(self.X.size())
