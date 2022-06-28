# Multi-GPU Example
# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

# Name: Clare Heinbaugh
# Date: 6/28/2022

# Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches
# and run the computation for each of the smaller mini-batches in parallel.

# Data Parallelism is implemented using torch.nn.DataParallel. One can wrap a Module in
# DataParallel and it will be parallelized over multiple GPUs in the batch dimension.

import torch
import torch.nn as nn

class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)

        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)

        self.block3 = nn.Linear(20, 20)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x



