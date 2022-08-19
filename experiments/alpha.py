import numpy as np

from utils import create_shell_files

if __name__ == "__main__":
    hosts = np.array(
        [2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    )
    all_scripts = []

    default_scripts_dict = {
        # MNIST
        "fed_avg_dataset=mnist": "python3 ../main.py --algorithm fedavg --dataset mnist --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --local_LR 0.001 --use_adam 1 --should_log 1",
        "one_shot_dataset=mnist": "python3 ../main.py --algorithm oneshot --dataset mnist --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --one_shot_sampling all --use_adam 1 --should_log 1",
        "one_fed_vae_dataset=mnist": "python3 ../main.py --algorithm onefedvae --dataset mnist --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_train_samples 5000 --classifier_epochs 10 --use_adam 1 --should_log 1",
        "fed_vae_dataset=mnist": "python3 ../main.py --algorithm fedvae --dataset mnist --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --z_dim 10 --beta 1.0 --classifier_train_samples 5000 --classifier_epochs 10 --decoder_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --should_log 1",
        # FashionMNIST
        "fed_avg_dataset=fashion": "python3 ../main.py --algorithm fedavg --dataset fashion --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 10 --local_LR 0.001 --use_adam 1 --should_log 1",
        "one_shot_dataset=fashion": "python3 ../main.py --algorithm oneshot --dataset fashion --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --one_shot_sampling data --K 7 --use_adam 1 --should_log 1",
        "one_fed_vae_dataset=fashion": "python3 ../main.py --algorithm onefedvae --dataset fashion --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --z_dim 100 --beta 1.0 --classifier_train_samples 5000 --classifier_epochs 5 --use_adam 1 --should_log 1",
        "fed_vae_dataset=fashion": "python3 ../main.py --algorithm fedvae --dataset fashion --num_users 10 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --local_LR 0.001 --z_dim 100 --beta 1.0 --classifier_train_samples 5000 --classifier_epochs 5 --decoder_train_samples 5000 --decoder_epochs 7 --decoder_LR 0.01 --use_adam 1 --should_log 1",
    }

    all_seeds = [1588, 1693, 7089, 4488, 3776]

    alphas = [0.1, 0.01, 0.001]

    for seed in all_seeds:
        for alpha in alphas:
            for default_script_name in default_scripts_dict.keys():
                cur_name = f"runs/{default_script_name}_alpha={alpha}_seed={seed}"
                script = default_scripts_dict.get(default_script_name)
                all_scripts.append(
                    f"{script} --alpha {alpha} --seed {seed} --cur_run_name {cur_name}"
                )

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "alpha")
