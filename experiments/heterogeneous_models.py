import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array(
        [2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    )
    all_scripts = []
    datasets = ["mnist", "fashion"]
    algorithms = ["fedvae", "onefedvae"]
    default_script = f"python3 ../main.py --should_log 1 --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1"
    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    experiments = ["--heterogeneous_models_exp=0", "--heterogeneous_models_exp=1"]

    all_seeds = [1588, 1693, 7089, 4488, 3776]

    for seed in all_seeds:
        for exp in experiments:
            for default_script_name in default_scripts_dict:
                cur_script = default_scripts_dict.get(default_script_name)
                cur_name = f"runs/{default_script_name}_exp={exp.replace('--', '')}_seed={seed}"
                all_scripts.append(
                    f"{cur_script} {exp.replace('=', ' ')} --seed {seed} --cur_run_name {cur_name}"
                )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "heterogeneous")
