import socket

import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array([19, 20, 21, 22, 23, 24])
    host_num = int(socket.gethostname().split("-")[1])
    host_num_index = int(np.where(hosts == host_num)[0])

    shell_file_name = f"experiments/hyperparam_runs_{host_num}.sh"
    print(host_num)

    beta_vals = [0.1, 0.5, 1.0]
    local_lr_vals = [0.01, 0.001, 0.0001]

    all_scripts = []
    for beta in beta_vals:
        for local_lr in local_lr_vals:
            all_scripts.append(
                f"python3 ../main.py --algorithm fedvae --dataset mnist --num_users 10 --alpha 1.0 --sample_ratio 0.1 --glob_epochs 3 --local_epochs 3 --should_log 1 --z_dim 50 --beta {beta} --local_LR {local_lr}"
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
