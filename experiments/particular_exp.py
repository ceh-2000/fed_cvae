import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array(
        [2, 4, 5, 6, 7, 8]
    )
    all_scripts = [
        "python3 main.py --algorithm fedavg --dataset svhn --num_users 10 --alpha 100.0 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --should_log 1 --local_LR 0.001 --use_adam 1",
        "python3 main.py --algorithm fedavg --dataset cifar10 --num_users 10 --alpha 100.0 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --should_log 1 --local_LR 0.001 --use_adam 1",
        "python3 main.py --algorithm fedavg --dataset svhn --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --should_log 1 --local_LR 0.001 --use_adam 1",
        "python3 main.py --algorithm fedavg --dataset cifar10 --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --should_log 1 --local_LR 0.001 --use_adam 1",
        "python3 main.py --algorithm fedvae --dataset svhn --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --should_log 1 --z_dim 500 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 5 --decoder_num_train_samples 5000 --decoder_epochs 10 --decoder_LR 0.01 --use_adam 1",
        "python3 main.py --algorithm fedvae --dataset cifar10 --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --should_log 1 --z_dim 500 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 5 --decoder_num_train_samples 5000 --decoder_epochs 10 --decoder_LR 0.01 --use_adam 1"
    ]

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "particular_exp")
