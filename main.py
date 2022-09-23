import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from data import Data
from servers.server_fed_avg import ServerFedAvg
from servers.server_fed_vae import ServerFedVAE
from servers.server_one_fed_vae import ServerOneFedVAE
from servers.server_one_shot import ServerOneShot
from unachievable_ideal import UnachievableIdeal


def run_job(args):
    torch.manual_seed(args.seed)

    writer = None

    if args.should_log:
        if args.cur_run_name == "":
            # Before logging anything, we need to create a SummaryWriter instance.
            # Writer will output to ./runs/ directory by default.
            cur_run_name = f"runs/algorithm={args.algorithm}_dataset={args.dataset}_users={args.num_users}_local_epochs={args.local_epochs}_local_LR={args.local_LR}_alpha={args.alpha}_sample_ratio={args.sample_ratio}"

            # Adding in algo-specific hyperparams
            if args.algorithm == "fedavg":
                cur_run_name = cur_run_name + f"_glob_epochs={args.glob_epochs}"
            elif args.algorithm == "oneshot":
                cur_run_name = (
                    cur_run_name
                    + f"_sampling={args.one_shot_sampling}_K={args.K if args.one_shot_sampling != 'all' else args.num_users}"
                )
            elif args.algorithm == "onefedvae":
                cur_run_name = (
                    cur_run_name
                    + f"_z_dim={args.z_dim}_beta={args.beta}_classifier_train_samples={args.classifier_num_train_samples}_classifier_epochs={args.classifier_epochs}_uniform_range={args.uniform_range}"
                )
            elif args.algorithm == "fedvae":
                cur_run_name = (
                    cur_run_name
                    + f"_z_dim={args.z_dim}_classifier_train_samples={args.classifier_num_train_samples}_classifier_epochs={args.classifier_epochs}_decoder_train_samples={args.decoder_num_train_samples}_decoder_epochs={args.decoder_epochs}_decoder_LR={args.decoder_LR}_uniform_range={args.uniform_range}"
                )
            elif args.algorithm == "central":
                cur_run_name = f"runs/algorithm=central_model_sampling_ratio={args.sample_ratio}_number_of_epochs={args.glob_epochs}"
        else:
            cur_run_name = args.cur_run_name

        writer = SummaryWriter(log_dir=cur_run_name)

    # Get the data
    d = Data(
        args.data_seed,
        args.dataset,
        args.num_users,
        writer,
        args.algorithm == "central",
        alpha=args.alpha,
        sample_ratio=args.sample_ratio,
        visualize=True if args.alpha is not None else False,
    )

    # Train in a centralized manner to generate the "unachievable ideal"
    if args.algorithm == "central":
        params = {
            "device": device,
            "glob_epoch": args.glob_epochs,
            "train_data": d.train_data,
            "test_data": d.test_data,
            "num_channels": d.num_channels,
            "num_classes": d.num_classes,
            "writer": writer,
        }
        i = UnachievableIdeal(params)

        i.train()
        i.test()

    # Distribute training across user devices
    else:
        # Initialize the server which manages all users
        default_params = {
            "device": device,
            "num_users": args.num_users,
            "user_fraction": args.user_fraction,
            "glob_epochs": args.glob_epochs,
            "local_epochs": args.local_epochs,
            "local_LR": args.local_LR,
            "use_adam": args.use_adam,
            "dataset_name": args.dataset,
            "data_subsets": d.train_data,
            "data_server": d.test_data,
            "num_channels": d.num_channels,
            "num_classes": d.num_classes,
            "writer": writer,
        }

        if args.algorithm == "fedavg":
            s = ServerFedAvg(default_params)
        elif args.algorithm == "oneshot":
            s = ServerOneShot(
                default_params,
                args.one_shot_sampling,
                args.user_data_split,
                args.K,
                args.should_initialize_same_exp,
            )
        elif args.algorithm == "onefedvae":
            s = ServerOneFedVAE(
                default_params,
                args.z_dim,
                d.image_size,
                args.beta,
                args.classifier_num_train_samples,
                args.classifier_epochs,
                args.uniform_range,
                args.should_weight_exp,
                args.should_initialize_same_exp,
                args.heterogeneous_models_exp,
            )
        elif args.algorithm == "fedvae":
            s = ServerFedVAE(
                default_params,
                args.z_dim,
                d.image_size,
                args.beta,
                args.classifier_num_train_samples,
                args.classifier_epochs,
                args.decoder_num_train_samples,
                args.decoder_epochs,
                args.decoder_LR,
                args.uniform_range,
                args.should_weight_exp,
                args.should_initialize_same_exp,
                args.should_avg_exp,
                args.should_fine_tune_exp,
                args.heterogeneous_models_exp,
                args.transform_exp,
            )
        else:
            raise NotImplementedError(
                "The specified algorithm has not been implemented."
            )

        s.create_users()

        print(f"_________________________________________________\n\n")

        s.train()
        s.test()

    if args.should_log:
        # Make sure that all pending events have been written to disk
        writer.flush()

        # Close writer when finished
        writer.close()


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(float, strings.split(","))
    return tuple(mapped_int)


if __name__ == "__main__":
    ####################################################################################################################
    # Parse command line arguments
    ####################################################################################################################

    # Extract command line arguments
    parser = argparse.ArgumentParser()

    # General command line arguments for all models
    parser.add_argument(
        "--algorithm",
        type=str,
        help="Name of algorithm to use to train global model",
        default="fedavg",
    )
    parser.add_argument("--dataset", type=str, help="Name of dataset", default="mnist")
    parser.add_argument(
        "--num_users",
        type=int,
        help="Number of users to divide data between",
        default=10,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Measure of heterogeneity (higher is more homogeneous, lower is more heterogenous)",
        default=None,
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        help="Fraction of training data to make available to users",
        default=1,
    )
    parser.add_argument(
        "--glob_epochs",
        type=int,
        help="Number of global epochs server model should train for",
        default=3,
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        help="Number of local epochs users should train for",
        default=5,
    )
    parser.add_argument(
        "--should_log",
        type=int,
        help="Turn logging to tensorboard on/off",
        default=0,
    )
    parser.add_argument(
        "--seed", type=int, help="Seed to ensure same model training", default=1693
    )
    parser.add_argument(
        "--data_seed", type=int, help="Seed to ensure same data split", default=1693
    )
    parser.add_argument(
        "--user_fraction",
        type=float,
        default=1.0,
        help="Fraction of users that we should sample each round",
    )
    parser.add_argument(
        "--local_LR",
        type=float,
        default=0.001,
        help="Local (user) learning rate (either for classifier or CVAE)",
    )
    parser.add_argument(
        "--use_adam",
        type=int,
        default=1,
        help="If 1, use Adam as the local optimizer, else use SGD",
    )

    # Command line arguments for specific models
    parser.add_argument(
        "--one_shot_sampling",
        type=str,
        help="Method to sample users for one shot ensembling",
        default="random",
    )
    parser.add_argument(
        "--user_data_split",
        type=float,
        help="The ratio of training to validation data for users",
        default=0.9,
    )
    parser.add_argument(
        "--K",
        type=int,
        help="Number of users to select for one shot ensembling",
        default=2,
    )
    parser.add_argument(
        "--z_dim", type=int, help="Latent vector dimension for VAE", default=50
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Weight on the KL divergence term for FedVAE",
        default=1.0,
    )
    parser.add_argument(
        "--classifier_num_train_samples",
        type=int,
        help="Number of images and labels to generate for server classifier training",
        default=1000,  # For MNIST, that's ~100 per class
    )
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        help="Number of epochs to train classifier in server",
        default=5,
    )
    parser.add_argument(
        "--decoder_num_train_samples",
        type=int,
        help="Number of images and labels to generate for server decoder KD fine-tuning",
        default=1000,
    )
    parser.add_argument(
        "--decoder_epochs",
        type=int,
        help="Number of epochs to fine-tune the server decoder for",
        default=5,
    )
    parser.add_argument(
        "--decoder_LR",
        type=float,
        default=0.001,
        help="Learning rate to use for decoder KD fine-tuning",
    )
    parser.add_argument(
        "--uniform_range",
        type=tuple_type,
        default=(-1.0, 1.0),
        help="Range of values to use when sampling from the decoder's latent space.",
    )

    # Command line arguments for experiments
    parser.add_argument(
        "--cur_run_name",
        type=str,
        default="",
        help="Modify what the run is named in Tensorboard",
    )
    parser.add_argument(
        "--should_weight_exp",
        type=int,
        default=1,
        help="Whether or not to weight server decoder aggregation and sampling",
    )
    parser.add_argument(
        "--should_initialize_same_exp",
        type=int,
        default=0,
        help="Whether or not to initialize all user models the same",
    )
    parser.add_argument(
        "--should_avg_exp",
        type=int,
        default=0,
        help="Whether or not to average server decoder",
    )
    parser.add_argument(
        "--should_fine_tune_exp",
        type=int,
        default=1,
        help="Whether or not fine tune server decoder",
    )
    parser.add_argument(
        "--heterogeneous_models_exp",
        type=str,
        default="0",
        help="Whether or not to give users heterogeneous models. Provide which versions of the CVAE to add as a string.",
    )
    parser.add_argument(
        "--transform_exp",
        type=int,
        default=0,
        help="Whether or not to apply transforms to the training images generated in the FedVAE pipeline.",
    )

    args = parser.parse_args()
    args.should_log = bool(args.should_log)
    args.use_adam = bool(args.use_adam)

    # Get available gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####################################################################################################################
    # Output model arguments
    ####################################################################################################################

    print("_________________________________________________\n")
    print("GENERAL COMMAND LINE ARGUMENTS")
    print()
    print("Using device:", device)
    print("Dataset name:", args.dataset)
    print("Algorithm:", args.algorithm)
    print("Portion of the dataset used:", args.sample_ratio)
    print("Logging?", "yes" if args.should_log else "no")
    print("Model seed:", args.seed)
    print("Data seed:", args.data_seed)

    # FedAvg-specific parameters
    if args.algorithm == "fedavg":
        print(f"Using {'Adam' if args.use_adam else 'SGD'} as the local optimizer")
        print(
            "Level of heterogeneity (alpha):",
            args.alpha if args.alpha is not None else "perfectly homogeneous",
        )
        print("Number of users for training:", args.num_users)
        print("Number of local epochs:", args.local_epochs)
        print("Local learning rate:", args.local_LR)
        print("Number of global epochs:", args.glob_epochs)
        print(
            "Fraction of users sampled for each communication round:",
            args.user_fraction,
        )

    # OneShot-specific parameters
    elif args.algorithm == "oneshot":
        print(f"Using {'Adam' if args.use_adam else 'SGD'} as the local optimizer")
        print(
            "Level of heterogeneity (alpha):",
            args.alpha if args.alpha is not None else "perfectly homogeneous",
        )
        print("Number of users for training:", args.num_users)
        print("Number of local epochs:", args.local_epochs)
        print("Local learning rate:", args.local_LR)

        args.glob_epochs = 1
        print("Number of global epochs:", 1)
        print("One shot sampling method:", args.one_shot_sampling)
        print(
            "Portion of data used for training:",
            args.user_data_split if args.one_shot_sampling == "validation" else 1,
        )
        print(
            "Number of users to select for one shot ensembling:",
            args.K if args.one_shot_sampling != "all" else "all",
        )

    # OneFedVAE-specific parameters
    elif args.algorithm == "onefedvae":
        print(f"Using {'Adam' if args.use_adam else 'SGD'} as the local optimizer")
        print(
            "Level of heterogeneity (alpha):",
            args.alpha if args.alpha is not None else "perfectly homogeneous",
        )
        print("Number of users for training:", args.num_users)
        print("Number of local epochs:", args.local_epochs)
        print("Local learning rate:", args.local_LR)

        args.glob_epochs = 1
        print("Number of global epochs:", 1)

        print("Latent vector dimension for VAE:", args.z_dim)
        print("Weight on the KL divergence term (beta):", args.beta)
        print(
            "Number of samples to generate for server classifier training:",
            args.classifier_num_train_samples,
        )
        print(
            "Number of epochs to train classifier in server:",
            args.classifier_epochs,
        )
        print("Range of values to use for decoder sample:", args.uniform_range)

    # FedVAE-specific parameters
    elif args.algorithm == "fedvae":
        print(f"Using {'Adam' if args.use_adam else 'SGD'} as the local optimizer")
        print(
            "Level of heterogeneity (alpha):",
            args.alpha if args.alpha is not None else "perfectly homogeneous",
        )
        print("Number of users for training:", args.num_users)
        print("Number of local epochs:", args.local_epochs)
        print("Local learning rate:", args.local_LR)
        print("Number of global epochs:", args.glob_epochs)
        print(
            "Fraction of users sampled for each communication round:",
            args.user_fraction,
        )
        print("Latent vector dimension for VAE:", args.z_dim)
        print("Weight on the KL divergence term (beta):", args.beta)
        print(
            "Number of samples to generate for server classifier training:",
            args.classifier_num_train_samples,
        )
        print(
            "Number of epochs to train classifier in server:",
            args.classifier_epochs,
        )
        print(
            "Number of images and labels to generate for server decoder KD fine-tuning:",
            args.decoder_num_train_samples,
        )
        print(
            "Number of epochs to fine-tune the server decoder for:",
            args.decoder_epochs,
        )
        print(
            "Learning rate for server decoder fine-tuning:",
            args.decoder_LR,
        )
        print("Range of values to use for decoder sample:", args.uniform_range)

    # Centralized model-specific parameters
    elif args.algorithm == "central":
        print(f"Using {'Adam' if args.use_adam else 'SGD'} as the local optimizer")
        print("Number of epochs:", args.glob_epochs)

    print()

    ####################################################################################################################
    # Experiment information
    ####################################################################################################################

    if args.cur_run_name != "":
        print(
            "Should we weight the server decoder aggregation and sampling?",
            "yes" if args.should_weight_exp else "no",
        )
        print(
            "Should initialize user models the same?",
            "yes" if args.should_initialize_same_exp else "no",
        )
        print(
            "Should we average the server decoder?",
            "yes" if args.should_avg_exp else "no",
        )
        print(
            "Should we fine tune the server decoder?",
            "yes" if args.should_fine_tune_exp else "no",
        )
        print(
            "Should we use heterogeneous models?",
            "yes" if len(args.heterogeneous_models_exp) > 1 else "no",
        )
        print(
            "Should we apply transforms?",
            "yes" if args.transform_exp == 1 else "no",
        )

    print("_________________________________________________\n")

    run_job(args)
