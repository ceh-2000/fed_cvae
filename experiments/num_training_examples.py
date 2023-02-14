import random

import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([7, 8, 9, 10, 12])

    all_scripts = []
    datasets = ["mnist", "fashion", "svhn"]
    algorithms = ["fedavg", "oneshot", "fedcvaekd", "fedcvaeens"]
    default_script = (
        f"python3 ../main.py --should_log 1 --glob_epochs 1 --alpha 0.01 --num_users 10"
    )

    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    sample_ratio_vals = [0.01, 0.05, 0.10, 0.25, 0.50]
    all_seeds = [1588, 1693, 7089, 4488, 3776]

    for seed in all_seeds:
        for sample_ratio in sample_ratio_vals:
            for default_script_name in default_scripts_dict.keys():
                cur_name = f"runs/{default_script_name}_sample_ratio={sample_ratio}_seed={seed}"
                script = default_scripts_dict.get(default_script_name)
                part_1, part_2 = script.split(" --sample_ratio ")
                script = f"{part_1.strip()} --sample_ratio {sample_ratio} {part_2.strip()[4:]}"

                all_scripts.append(f"{script} --seed {seed} --cur_run_name {cur_name}")

    print("Number of experiments:", len(all_scripts))

    random.shuffle(all_scripts)

    create_shell_files(all_scripts, hosts, "sample_ratio")
