import argparse

import torch

from data import Data
from server import Server


def get_gpus():
    num_of_gpus = torch.cuda.device_count()

    devs = []
    for d in range(num_of_gpus):
        devs.append(f"cuda:{d}")

    return devs


if __name__ == "__main__":
    # Extract command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--glob_epochs", type=int, default=10)
    args = parser.parse_args()

    # Get available gpus
    devices = get_gpus()

    print("Number of devices: ", len(devices))
    print("Number of users for training: ", args.num_users)
    print("Number of local epochs: ", args.local_epochs)
    print("Number of global epochs: ", args.glob_epochs)

    # Get the data
    d = Data()

    # Server manages all users
    s = Server(devices=devices, num_users=args.num_users, glob_epochs=args.glob_epochs, local_epochs=args.local_epochs,
               X=d.X, y=d.y)

    s.create_users()

    s.train()
