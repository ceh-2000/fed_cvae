# Multi-GPU Example
# https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

# Name: Clare Heinbaugh
# Date: 6/28/2022

import argparse
import torch

def get_gpus():
    num_of_gpus = torch.cuda.device_count()

    devices = []
    for d in range(num_of_gpus):
        devices.append(f'cuda:{d}')

    return devices

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=10)
    args = parser.parse_args()

    devices = get_gpus()

    print('Number of devices: ', len(devices))
    print('Number of users for training: ', args.num_users)






#
# # Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches
# # and run the computation for each of the smaller mini-batches in parallel.
#
# # Data Parallelism is implemented using torch.nn.DataParallel. One can wrap a Module in
# # DataParallel and it will be parallelized over multiple GPUs in the batch dimension.
#
# import torch
# import torch.nn as nn
#
# class DataParallelModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.block1 = nn.Linear(10, 20)
#
#         # wrap block2 in DataParallel
#         self.block2 = nn.Linear(20, 20)
#         self.block2 = nn.DataParallel(self.block2)
#
#         self.block3 = nn.Linear(20, 20)
#
#         self.m = nn.Softmax(dim = 1)
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.m(x)
#
#         return x
#
# if __name__ == "__main__":
#     x = torch.rand(50, 10)
#     y = torch.randint(size=(50, 1), low=0, high=2)
#
#     print(y)
#
#     model = DataParallelModel()
#     logits = model(x)
#
#     print(logits)
#
#
#
#
#
#
#
