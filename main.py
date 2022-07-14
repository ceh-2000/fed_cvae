import argparse
import torch
from data import Data
from server import Server
from torch.utils.tensorboard import SummaryWriter


def get_gpus():
    num_of_gpus = torch.cuda.device_count()

    devs = []
    for d in range(num_of_gpus):
        devs.append(f"cuda:{d}")

    return devs


def run_job(args):
    # Get the data
    d = Data(args.dataset, args.num_users)

    for i in range(args.trials):
        torch.manual_seed(i)

        writer = None
        if args.should_log:
            # Before logging anything, we need to create a SummaryWriter instance.
            # Writer will output to ./runs/ directory by default.
            cur_run_name = f"runs/iter={i}_users={args.num_users}_glob_epochs={args.glob_epochs}_local_epochs={args.local_epochs}"
            writer = SummaryWriter(log_dir=cur_run_name)

        # Initialize the server which manages all users
        s = Server(
            devices=devices,
            num_users=args.num_users,
            glob_epochs=args.glob_epochs,
            local_epochs=args.local_epochs,
            data_subsets=d.train_data,
            data_server=d.test_data,
            num_channels=d.num_channels,
            num_classes=d.num_classes,
            writer=writer
        )
        s.create_users()

        print(
            f"[--------------------Starting training iteration {i}--------------------]"
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
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--glob_epochs", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--should_log", type=bool, default=False)
    args = parser.parse_args()

    # Get available gpus
    devices = get_gpus()

    print("Number of devices: ", len(devices))
    print("Dataset name: ", args.dataset)
    print("Number of trials: ", args.trials)
    print("Number of users for training: ", args.num_users)
    print("Number of local epochs: ", args.local_epochs)
    print("Number of global epochs: ", args.glob_epochs)
    print("Logging? ", args.should_log)
    print("_________________________________________________\n\n")

    run_job(args)
