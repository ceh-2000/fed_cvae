import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([19, 20, 21])
    all_scripts = [
        "python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --algorithm fedvae --dataset mnist --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 10 --decoder_num_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --alpha 0.01 --seed 1588 --cur_run_name runs/few_shot_seed=1588",
        "python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --algorithm fedvae --dataset mnist --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 10 --decoder_num_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --alpha 0.01 --seed 1588 --cur_run_name runs/few_shot_seed=1588",
        "python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --algorithm fedvae --dataset mnist --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 10 --decoder_num_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --alpha 0.01 --seed 1588 --cur_run_name runs/few_shot_seed=1588",
        "python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --algorithm fedvae --dataset mnist --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 10 --decoder_num_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --alpha 0.01 --seed 1588 --cur_run_name runs/few_shot_seed=1588",
        "python3 ../main.py --should_log 1 --num_users 10 --glob_epochs 1 --algorithm fedvae --dataset mnist --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --classifier_epochs 10 --decoder_num_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --alpha 0.01 --seed 1588 --cur_run_name runs/few_shot_seed=1588",
    ]

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "particular_exp")
