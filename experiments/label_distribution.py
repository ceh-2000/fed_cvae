import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([19, 20, 21, 22, 23, 24, 7, 8, 10, 12])

    all_scripts = []
    datasets = ["mnist", "fashion", "svhn"]
    algorithms = ["fedvae", "onefedvae"]
    default_script = (
        f"python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --alpha 0.01"
    )

    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    # Noisy experiments
    noise_seeds_vals = [1588, 1693, 7089, 4488, 3776]
    noise_weight_vals = [0.05, 0.01, 0.001]

    for noise_seed in noise_seeds_vals:
        for noise_weight in noise_weight_vals:
            for default_script_name in default_scripts_dict.keys():
                cur_name = f"runs/{default_script_name}_noise_weight={noise_weight}_noise_seed={noise_seed}"
                script = default_scripts_dict.get(default_script_name)
                all_scripts.append(
                    f"{script} --noisy_label_dists noisy --noise_weight {noise_weight} --noise_seed {noise_seed} --cur_run_name {cur_name}"
                )

    # Uniform experiments
    all_seeds = [1588, 1693, 7089, 4488, 3776]
    noise_weight = -1.0  # Noise weight of -1 indicates uniform

    for seed in all_seeds:
        for default_script_name in default_scripts_dict.keys():
            cur_name = (
                f"runs/{default_script_name}_noise_weight={noise_weight}_seed={seed}"
            )
            script = default_scripts_dict.get(default_script_name)
            all_scripts.append(
                f"{script} --noisy_label_dists uniform --seed {seed} --cur_run_name {cur_name}"
            )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "label_distribution")
