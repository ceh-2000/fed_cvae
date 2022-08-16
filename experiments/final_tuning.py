import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    )

    all_scripts = []

    # FedVAE tuning (use the same values for OneFedVAE)
    algorithm = "fedvae"
    num_users = 10
    glob_epochs = 1
    local_epochs = 15
    alpha = 0.01
    sample_ratio = 5
    beta = 1.0
    classifier_num_train_samples = 5000
    decoder_num_train_samples = 5000
    use_adam = 1
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
                            f"python3 ../main.py --should_log 1 --algorithm {algorithm} --dataset {dataset} "
                            f"--num_users {num_users} --alpha {alpha} --sample_ratio {sample_ratio} "
                            f"--glob_epochs {glob_epochs} --local_epochs {local_epochs} --z_dim {z_dim} "
                            f"--beta {beta} --classifier_num_train_samples {classifier_num_train_samples} "
                            f"--decoder_num_train_samples {decoder_num_train_samples} "
                            f"--classifier_epochs {classifier_num_epochs} --decoder_epochs {decoder_num_epochs} "
                            f"--local_LR {local_LR} --decoder_LR {decoder_LR}"
                        )

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"hyperparam_runs_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
