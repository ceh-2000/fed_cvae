import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    )

    all_scripts = []

    # FedVAE tuning (use the same values for OneFedVAE)
    algorithm = "fedvae"
    num_users = 10
    glob_epochs = 1
    local_epochs = 15
    alpha = 0.01
    sample_ratio = 0.5
    beta = 1.0
    classifier_num_train_samples = 5000
    decoder_num_train_samples = 5000
    use_adam = 1
    should_log = 1
    local_LR = 0.001
    decoder_LR = 0.01

    dataset_vals = ["mnist", "fashionmnist"]
    z_dim_vals = [10, 50, 100]
    decoder_num_epochs_vals = [3, 5, 7]
    classifier_num_epochs_vals = [5, 10, 15]
    should_weight_vals = [0, 1]

    for dataset in dataset_vals:
        for z_dim in z_dim_vals:
            for decoder_num_epochs in decoder_num_epochs_vals:
                for classifier_num_epochs in classifier_num_epochs_vals:
                    for should_weight in should_weight_vals:
                        all_scripts.append(
                            f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} "
                            f"--algorithm {algorithm} --dataset {dataset} --num_users {num_users} --alpha {alpha} "
                            f"--sample_ratio {sample_ratio} --glob_epochs {glob_epochs} --local_epochs {local_epochs} "
                            f"--z_dim {z_dim} --beta {beta} "
                            f"--classifier_num_train_samples {classifier_num_train_samples} "
                            f"--decoder_num_train_samples {decoder_num_train_samples} "
                            f"--classifier_epochs {classifier_num_epochs} --decoder_epochs {decoder_num_epochs} "
                            f"--local_LR {local_LR} --decoder_LR {decoder_LR} --should_weight {should_weight}"
                        )

    # One-shot FL tuning
    algorithm = "oneshot"
    num_users = 10
    glob_epochs = 1
    alpha = 0.01
    sample_ratio = 0.5
    user_data_split = 0.8
    use_adam = 1
    should_log = 1

    dataset_vals = ["mnist", "fashionmnist"]
    local_epochs_vals = [5, 10, 15]
    local_LR_vals = [0.01, 0.001, 0.0001]
    K_vals = [3, 5, 7]

    # one_shot_sampling_vals = ["random", "data", "validation", "all"]
    one_shot_sampling_vals = ["random", "data"]
    for dataset in dataset_vals:
        for local_epochs in local_epochs_vals:
            for local_LR in local_LR_vals:
                for one_shot_sampling in one_shot_sampling_vals:
                    for K in K_vals:
                        all_scripts.append(
                            f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} --algorithm {algorithm} "
                            f"--num_users {num_users} --glob_epochs {glob_epochs} --local_epochs {local_epochs} "
                            f"--alpha {alpha} --sample_ratio {sample_ratio} --dataset {dataset} --local_LR {local_LR} "
                            f"--one_shot_sampling {one_shot_sampling} --K {K}"
                        )

    one_shot_sampling = "validation"
    for dataset in dataset_vals:
        for local_epochs in local_epochs_vals:
            for local_LR in local_LR_vals:
                for K in K_vals:
                    all_scripts.append(
                        f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} --algorithm {algorithm} "
                        f"--num_users {num_users} --glob_epochs {glob_epochs} --local_epochs {local_epochs} --alpha {alpha} "
                        f"--sample_ratio {sample_ratio} --one_shot_sampling {one_shot_sampling} "
                        f"--user_data_split {user_data_split} --dataset {dataset} --local_LR {local_LR} --K {K}"
                    )

    one_shot_sampling = "all"
    for dataset in dataset_vals:
        for local_epochs in local_epochs_vals:
            for local_LR in local_LR_vals:
                all_scripts.append(
                    f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} --algorithm {algorithm} "
                    f"--num_users {num_users} --glob_epochs {glob_epochs} --local_epochs {local_epochs} --alpha {alpha} "
                    f"--sample_ratio {sample_ratio} --one_shot_sampling {one_shot_sampling} --dataset {dataset}"
                    f"--local_LR {local_LR}"
                )

    # FedAvg Tuning
    algorithm = "fedavg"
    num_users = 10
    glob_epochs = 1
    alpha = 0.01
    sample_ratio = 0.5
    use_adam = 1
    should_log = 1

    dataset_vals = ["mnist", "fashionmnist"]
    local_LR_vals = [0.01, 0.001, 0.0001]
    local_epochs_vals = [5, 7, 10]

    for dataset in dataset_vals:
        for local_LR in local_LR_vals:
            for local_epochs in local_epochs_vals:
                all_scripts.append(
                    f"python3 ../main.py --should_log {should_log} --use_adam {use_adam} --algorithm {algorithm} "
                    f"--num_users {num_users} --glob_epochs {glob_epochs} --alpha {alpha} "
                    f"--sample_ratio {sample_ratio} --dataset {dataset} --local_LR {local_LR} "
                    f"--local_epochs {local_epochs}"
                )

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"hyperparam_runs_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
