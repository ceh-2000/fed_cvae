import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([19, 20, 21, 22, 23, 24])
    all_scripts = []
    datasets = ["mnist", "fashion", "svhn"]
    algorithms = ["fedvae"]
    default_script = f"python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1"
    default_scripts_dict = populate_default_scripts(
        datasets, algorithms, default_script
    )

    experiments = [
        "--should_weight_exp=1 --should_initialize_same_exp=0 --should_avg_exp=1 --should_fine_tune_exp=0",
    ]
    all_seeds = [1588, 1693, 7089, 4488, 3776]
    alpha_vals = [0.05, 0.01, 0.001]
    num_epochs_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    classifier_num_train_samples = 1000

    for seed in all_seeds:
        for num_epochs in num_epochs_vals:
            for alpha in alpha_vals:
                for exp in experiments:
                    for default_script_name in default_scripts_dict:
                        cur_script = default_scripts_dict.get(default_script_name)
                        part_1, part_2 = cur_script.split(" --local_epochs ")
                        cur_loc_epochs = round(num_epochs * int(part_2[:2]))
                        part_2, part_3 = part_2.split(
                            " --classifier_num_train_samples "
                        )

                        cur_name = f"runs/{default_script_name}_exp={exp.replace('--', '').replace(' ', '_')}_local_epochs={cur_loc_epochs}_alpha={alpha}_seed={seed}"
                        all_scripts.append(
                            f"{part_1.strip()} --local_epochs {cur_loc_epochs}{part_2.strip()[2:]} --classifier_num_train_samples {classifier_num_train_samples}{part_3.strip()[4:]} {exp.replace('=', ' ')} --alpha {alpha} --seed {seed} --cur_run_name {cur_name}"
                        )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "ablation")
