import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([1, 2])
    all_scripts = []
    datasets = ["fashion"]
    algorithms = ["oneshot"]
    default_script = f"python3 ../main.py --should_log 1 --sample_ratio 0.5 --glob_epochs 1 --alpha 0.01"

    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    num_users = [5, 10, 20, 50]
    all_seeds = [1588, 1693, 7089, 4488, 3776]

    for seed in all_seeds:
        for u in num_users:
            for default_script_name in default_scripts_dict.keys():
                cur_name = f"runs/{default_script_name}_num_users={u}_seed={seed}"
                script = default_scripts_dict.get(default_script_name)
                all_scripts.append(
                    f"{script} --num_users {u} --seed {seed} --cur_run_name {cur_name}"
                )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "num_users")
