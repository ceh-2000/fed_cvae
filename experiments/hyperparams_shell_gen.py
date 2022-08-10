import socket

import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    )
    host_num = int(socket.gethostname().split("-")[1])
    host_num_index = int(np.where(hosts == host_num)[0])

    shell_file_name = f"hyperparam_runs_{host_num}.sh"
    print(host_num)

    beta_vals = [0.1, 0.5, 1.0]
    classifier_train_samples_vals = [1000, 5000]
    decoder_train_samples_vals = [1000, 5000]
    classifier_num_epochs_vals = [5, 10]
    decoder_num_epochs_vals = [5, 10]
    local_lr_vals = [0.01, 0.001, 0.0001]
    decoder_lr_vals = [0.01, 0.001, 0.0001, 0.00001]

    all_scripts = []
    for beta in beta_vals:
        for classifier_train_samples in classifier_train_samples_vals:
            for decoder_train_samples in decoder_train_samples_vals:
                for classifier_num_epochs in classifier_num_epochs_vals:
                    for decoder_num_epochs in decoder_num_epochs_vals:
                        for local_lr in local_lr_vals:
                            for decoder_lr in decoder_lr_vals:
                                all_scripts.append(
                                    f"python3 ../main.py --algorithm fedvae --dataset mnist --num_users 10 --alpha "
                                    f"1.0 --sample_ratio 0.5 --glob_epochs 5 --local_epochs 15 --should_log 1 --z_dim "
                                    f"10 --beta {beta} --classifier_num_train_samples {classifier_train_samples} "
                                    f"--decoder_num_train_samples {decoder_train_samples} --classifier_epochs "
                                    f"{classifier_num_epochs} --decoder_epochs {decoder_num_epochs} --local_LR "
                                    f"{local_lr} --decoder_LR {decoder_lr} "
                                )

    counter = 0
    cur_host_scripts = []
    for i in split(all_scripts, len(hosts)):
        if counter == host_num_index:
            cur_host_scripts = i
        counter += 1
    print(cur_host_scripts)

    with open(shell_file_name, "w") as f:
        for c in cur_host_scripts:
            f.write(c)
            f.write("\n")
