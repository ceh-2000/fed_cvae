import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array([2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20])

    all_scripts = [
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 15 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 50 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 15 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 15 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 15 --decoder_epochs 3 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 5 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 5 --decoder_epochs 5 --local_LR 0.001 --decoder_LR 0.01 --should_weight 1",
        "python3 ../main.py --should_log 1 --use_adam 1 --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 100 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 5 --local_LR 0.001 --decoder_LR 0.01 --should_weight 0",
    ]

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"benchmarking_runs_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
