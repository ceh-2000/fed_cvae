import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([1])
    all_scripts = []
    datasets = ["svhn"]
    algorithms = ["fedvae"]
    default_script = f"python3 ../main.py --should_log 1 --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1"
    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    experiments = ["--transform_exp=0", "--transform_exp=1"]

    all_seeds = [1588, 1693, 7089]

    for seed in all_seeds:
        for exp in experiments:
            for default_script_name in default_scripts_dict:
                cur_script = default_scripts_dict.get(default_script_name)
                cur_name = (
                    f"runs/{default_script_name}_{exp.replace('--', '')}_seed={seed}"
                )
                all_scripts.append(
                    f"{cur_script} {exp.replace('=', ' ')} --seed {seed} --cur_run_name {cur_name}"
                )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "particular_exp")
