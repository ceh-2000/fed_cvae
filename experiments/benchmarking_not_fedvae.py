import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array([18, 19, 20, 21, 22, 23, 24])

    alphas = [0.01, 1.0, 100.0]

    all_scripts = [
        # Centralized model
        "python3 main.py --algorithm central --dataset mnist --sample_ratio 0.5 --glob_epochs 20 --should_log 1",
        # FedAvg (local learning rate is 0.01)
        f"python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha {alphas[0]} --sample_ratio 0.5 --glob_epochs 5 --local_epochs 15 --local_LR 0.01 --should_log 1",
        f"python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha {alphas[1]} --sample_ratio 0.5 --glob_epochs 5 --local_epochs 15 --local_LR 0.01 --should_log 1",
        f"python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha {alphas[2]} --sample_ratio 0.5 --glob_epochs 5 --local_epochs 15 --local_LR 0.01 --should_log 1",
        # One-shot FedVAE (hyperparameters from tuning few-shot FedVAE)
        f"python3 main.py --algorithm onefedvae --dataset mnist --num_users 10 --alpha {alphas[0]} --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --should_log 1 --z_dim 10 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 10",
        f"python3 main.py --algorithm onefedvae --dataset mnist --num_users 10 --alpha {alphas[1]} --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --should_log 1 --z_dim 10 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 10",
        f"python3 main.py --algorithm onefedvae --dataset mnist --num_users 10 --alpha {alphas[2]} --sample_ratio 0.5 --local_epochs 15 --local_LR 0.001 --should_log 1 --z_dim 10 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 10",
    ]

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"benchmarking_runs_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
