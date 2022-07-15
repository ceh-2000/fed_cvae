import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from data import Data
from servers.server_fed_avg import ServerFedAvg
from servers.server_one_shot import ServerOneShot


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
    elif args.algorithm == "one_shot":
        s = ServerOneShot(default_params, args.one_shot_sampling, args.user_data_split)
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

    # General command line arguments for all models
    parser.add_argument("--algorithm", type=str, help="Name of algorithm to use to train global model", default="fedavg")
    parser.add_argument("--dataset", type=str, help="Name of dataset", default="mnist")
    parser.add_argument("--num_users", type=int, help="Number of users to divide data between", default=10)
    parser.add_argument("--alpha", type=float, help="Measure of heterogeneity (higher is more homogeneous, lower is more heterogenous)", default=None)
    parser.add_argument("--sample_ratio", type=float, help="Fraction of training data to make available to users", default=1)
    parser.add_argument("--glob_epochs", type=int, help="Number of global epochs server model should train for", default=3)
    parser.add_argument("--local_epochs", type=int, help="Number of local epochs users should train for", default=5)
    parser.add_argument("--should_log", type=bool, help="Turn logging to tensorboard on/off", default=False)
    parser.add_argument("--seed", type=int, help="Seed to ensure same results", default=1693)

    # Command line arguments for specific models
    parser.add_argument("--one_shot_sampling", type=str, help="Method to sample users for one shot ensembling", default="random")
    parser.add_argument("--user_data_split", type=float, help="The ratio of training to validation data for users", default=0.9)

    args = parser.parse_args()

    # Get available gpus
    devices = get_gpus()
    print("_________________________________________________\n")
    print("GENERAL COMMAND LINE ARGUMENTS")
    print()
    print("Number of devices:", len(devices))
    print("Dataset name:", args.dataset)
    print("Algorithm:", args.algorithm)
    print(
        "Level of heterogeneity (alpha):",
        args.alpha if args.alpha is not None else "perfectly homogeneous",
    )
    print("Portion of the dataset used:", args.sample_ratio)
    print("Number of users for training:", args.num_users)
    print("Number of local epochs:", args.local_epochs)
    print("Number of global epochs:", args.glob_epochs)
    print("Logging?", args.should_log)
    print("Seed:", args.seed)

    print()

    print("MODEL SPECIFIC COMMAND LINE ARGUMENTS")
    print("One shot sampling method:", args.one_shot_sampling)
    print("Ratio of user training to validation data:", args.user_data_split)

    print("_________________________________________________\n")

    run_job(args)
