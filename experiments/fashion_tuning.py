import random

import numpy as np

from utils import create_shell_files

if __name__ == "__main__":
    hosts = np.array([1])

    all_scripts = []

    # GENERAL PARAMS
    num_users = 10
    glob_epochs = 1
    alpha = 0.01
    sample_ratio = 0.5
    should_log = 1
    dataset_vals = ["fashion"]

    # FedVAE tuning (use the same values for OneFedVAE)
    algorithm = "fedvae"
    beta = 1.0
    use_adam = 1
    local_LR = 0.001
    decoder_LR = 0.01
    should_weight = 1
    transform_exp = 0
    uniform_range = (-2.0, 2.0)

    z_dim_vals = [10, 15, 20, 100]
    local_epochs_vals = [10, 15, 25]
    classifier_num_train_samples_vals = [5000]
    classifier_epochs_vals = [3, 5, 7]
    decoder_num_train_samples_vals = [5000]
    decoder_num_epochs_vals = [5, 7, 10]

    for dataset in dataset_vals:
        for local_epochs in local_epochs_vals:
            for z_dim in z_dim_vals:
                for classifier_num_train_samples in classifier_num_train_samples_vals:
                    for classifier_num_epochs in classifier_epochs_vals:
                        for decoder_num_train_samples in decoder_num_train_samples_vals:
                            for decoder_num_epochs in decoder_num_epochs_vals:
                                all_scripts.append(
                                    f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} "
                                    f"--algorithm {algorithm} --dataset {dataset} --num_users {num_users} --alpha {alpha} "
                                    f"--sample_ratio {sample_ratio} --glob_epochs {glob_epochs} --local_epochs {local_epochs} "
                                    f'--z_dim {z_dim} --beta {beta} --uniform_range "{uniform_range}" '
                                    f"--classifier_num_train_samples {classifier_num_train_samples} "
                                    f"--decoder_num_train_samples {decoder_num_train_samples} "
                                    f"--classifier_epochs {classifier_num_epochs} --decoder_epochs {decoder_num_epochs} "
                                    f"--local_LR {local_LR} --decoder_LR {decoder_LR} --should_weight {should_weight} "
                                    f"--transform_exp {transform_exp} --heterogeneous_models_exp 0"
                                )

    algorithm = "onefedvae"
    beta = 1.0
    use_adam = 1
    local_LR = 0.001
    should_weight = 1
    transform_exp = 0
    uniform_range = (-3.0, 3.0)

    z_dim_vals = [10, 15, 20, 100]
    local_epochs_vals = [10, 15, 25]
    classifier_num_train_samples_vals = [5000]
    classifier_epochs_vals = [3, 5, 7]

    for dataset in dataset_vals:
        for z_dim in z_dim_vals:
            for local_epochs in local_epochs_vals:
                for classifier_num_train_samples in classifier_num_train_samples_vals:
                    for classifier_num_epochs in classifier_epochs_vals:
                        all_scripts.append(
                            f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} "
                            f"--algorithm {algorithm} --dataset {dataset} --num_users {num_users} --alpha {alpha} "
                            f"--sample_ratio {sample_ratio} --glob_epochs {glob_epochs} --local_epochs {local_epochs} "
                            f'--z_dim {z_dim} --beta {beta} --uniform_range "{uniform_range}" '
                            f"--classifier_num_train_samples {classifier_num_train_samples} "
                            f"--classifier_epochs {classifier_num_epochs} "
                            f"--local_LR {local_LR} --should_weight {should_weight} "
                            f"--transform_exp {transform_exp} --heterogeneous_models_exp 0"
                        )

    print("Number of experiments:", len(all_scripts))

    random.shuffle(all_scripts)

    create_shell_files(all_scripts, hosts, "fashion_tuning")
