import socket

import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    shell_file_name = "experiments/hyperparam_runs.sh"

    hosts = np.array([19, 20, 21, 22, 23, 24])
    host_num = int(socket.gethostname().split("-")[1])
    host_num_index = int(np.where(hosts == host_num)[0])
    print(host_num)

    a_vals = [0.1, 0.2, 0.3]
    b_vals = [1, 2, 3]
    c_vals = [10, 20, 30]

    all_scripts = []
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                all_scripts.append(f"python3 ../main.py -a {a} --b {b} --c {c}")

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
