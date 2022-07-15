import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from data import Data
from servers.server_fed_avg import ServerFedAvg


def get_gpus():
    num_of_gpus = torch.cuda.device_count()

    devs = []
    for d in range(num_of_gpus):
        devs.append(f"cuda:{d}")

    return devs


def run_job(args):
    torch.manual_seed(args.seed)

    writer = None
    if args.should_log:
        # Before logging anything, we need to create a SummaryWriter instance.
        # Writer will output to ./runs/ directory by default.
        cur_run_name = f"runs/users={args.num_users}_glob_epochs={args.glob_epochs}_local_epochs={args.local_epochs}_alpha={args.alpha}_sample_ratio={args.sample_ratio}"
        writer = SummaryWriter(log_dir=cur_run_name)

    # Get the data
    d = Data(
        args.dataset,
        args.num_users,
        writer,
        alpha=args.alpha,
        sample_ratio=args.sample_ratio,
        visualize=True if args.alpha is not None else False,
    )

    # Initialize the server which manages all users
    default_params = {
        "devices": devices,
        "num_users": args.num_users,
        "glob_epochs": args.glob_epochs,
        "local_epochs": args.local_epochs,
        "data_subsets": d.train_data,
        "data_server": d.test_data,
        "num_channels": d.num_channels,
        "num_classes": d.num_classes,
        "writer": writer,
    }

    if args.algorithm == "fedavg":
        s = ServerFedAvg(default_params)
    else:
        raise NotImplementedError(
            "The specified algorithm has not been implemented."
        )

    s.create_users()

    print(
        f"_________________________________________________\n\n"
    )

    s.train()
    s.test()

    if args.should_log:
        # Make sure that all pending events have been written to disk.
        writer.flush()

        # Close writer when finished.
        writer.close()


if __name__ == "__main__":
    # Extract command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1693, help="random seed for model training")
    parser.add_argument("--algorithm", type=str, default="fedavg", help="which algorithm should be used?")
    parser.add_argument("--dataset", type=str, default="mnist", help="which dataset should be used?")
    parser.add_argument("--num_users", type=int, default=10, help="how many users should be used?")
    parser.add_argument("--user_fraction", type=float, default=1.0, help="what fraction of users should we sample each round?")
    parser.add_argument("--alpha", type=float, default=None, help="level of data heterogeneity across users")
    parser.add_argument("--sample_ratio", type=float, default=1, help="what portion of the dataset should be used")
    parser.add_argument("--glob_epochs", type=int, default=3, help="number of global communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="number of local epochs between communication rounds for users")
    parser.add_argument("--should_log", type=bool, default=False, help="should we log results to tensorboard?")
    args = parser.parse_args()

    # Get available gpus
    devices = get_gpus()

    print("Number of devices: ", len(devices))
    print("Dataset name: ", args.dataset)
    print("Algorithm: ", args.algorithm)
    print(
        "Level of heterogeneity (alpha):",
        args.alpha if args.alpha is not None else "perfectly homogeneous",
    )
    print("Portion of the dataset used:", args.sample_ratio)
    print("Number of users for training: ", args.num_users)
    print("Number of local epochs: ", args.local_epochs)
    print("Number of global epochs: ", args.glob_epochs)
    print("Logging? ", args.should_log)
    print("Seed: ", args.seed)
    print("_________________________________________________\n\n")

    run_job(args)
