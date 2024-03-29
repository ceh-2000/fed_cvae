import random

import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([1])

    all_scripts = []
    datasets = ["mnist", "fashion", "svhn"]
    algorithms = ["fedavg", "oneshot", "fedcvaeens", "fedcvaekd"]
    default_script = (
        f"python3 ../main.py --should_log 1 --glob_epochs 1 --alpha 0.05 --num_users 10"
    )

    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    num_epochs = [i for i in range(1, 21)]
    all_seeds = [1588, 1693, 7089, 4488, 3776]

    for seed in all_seeds:
        for epoch in num_epochs:
            for default_script_name in default_scripts_dict.keys():
                cur_name = (
                    f"runs/{default_script_name}_local_epochs={epoch}_seed={seed}"
                )
                script = default_scripts_dict.get(default_script_name)
                part_1, part_2 = script.split(" --local_epochs ")
                script = f"{part_1.strip()} --local_epochs {epoch}{part_2.strip()[2:]}"

                all_scripts.append(f"{script} --seed {seed} --cur_run_name {cur_name}")

    print("Number of experiments:", len(all_scripts))

    random.shuffle(all_scripts)

    create_shell_files(all_scripts, hosts, "local_epochs")
