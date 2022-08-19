import numpy as np

from utils import create_shell_files

if __name__ == "__main__":
    hosts = np.array(
        [2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    )
    all_scripts = []

    # By default we include weighting, averaging, and fine-tuning
    default_script = "python3 ../main.py --algorithm fedvae --dataset fashion --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_log 1 --use_adam 1"

    all_seeds = [1588, 1693, 7089, 4488, 3776]

    experiments = [
        "--should_weight_exp=1 --should_initialize_same_exp=1 --should_avg_exp=1 --should_fine_tune_exp=1",
        "--should_weight_exp=0 --should_initialize_same_exp=1 --should_avg_exp=1 --should_fine_tune_exp=1",
        "--should_weight_exp=1 --should_initialize_same_exp=0 --should_avg_exp=1 --should_fine_tune_exp=1",
        "--should_weight_exp=1 --should_initialize_same_exp=1 --should_avg_exp=0 --should_fine_tune_exp=1",
        "--should_weight_exp=1 --should_initialize_same_exp=1 --should_avg_exp=1 --should_fine_tune_exp=0",
        "--should_weight_exp=1 --should_initialize_same_exp=0 --should_avg_exp=0 --should_fine_tune_exp=1",
        "--should_weight_exp=0 --should_initialize_same_exp=0 --should_avg_exp=0 --should_fine_tune_exp=1",
    ]

    for exp in experiments:
        for seed in all_seeds:
            exp_name = exp.replace("--", "_").replace(" ", "")[1:]
            run_name = f"runs/fedvae_{exp_name}_seed={seed}"
            all_scripts.append(
                default_script
                + f" {exp.replace('=', ' ')} --seed {seed} --cur_run_name {run_name}"
            )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "ablation")
