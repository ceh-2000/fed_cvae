import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17] #, 18, 19, 20, 21, 22, 23, 24]
    )

    alpha_vals = [0.01, 1.0, 100.0]
    settings = [
        {
            "local_lr": 0.001,
            "beta": 1.0,
            "decoder_lr": 0.01,
            "classifier_train_samples": 5000,
            "classifier_num_epochs": 5,
            "decoder_train_samples": 5000,
            "decoder_num_epochs": 5,
        },
        {
            "local_lr": 0.001,
            "beta": 1.0,
            "decoder_lr": 0.01,
            "classifier_train_samples": 5000,
            "classifier_num_epochs": 10,
            "decoder_train_samples": 5000,
            "decoder_num_epochs": 5,
        },
        {
            "local_lr": 0.0001,
            "beta": 1.0,
            "decoder_lr": 0.001,
            "classifier_train_samples": 5000,
            "classifier_num_epochs": 10,
            "decoder_train_samples": 5000,
            "decoder_num_epochs": 5,
        },
        {
            "local_lr": 0.001,
            "beta": 1.0,
            "decoder_lr": 0.01,
            "classifier_train_samples": 5000,
            "classifier_num_epochs": 10,
            "decoder_train_samples": 1000,
            "decoder_num_epochs": 5,
        },
        {
            "local_lr": 0.001,
            "beta": 1.0,
            "decoder_lr": 0.01,
            "classifier_train_samples": 5000,
            "classifier_num_epochs": 10,
            "decoder_train_samples": 5000,
            "decoder_num_epochs": 10,
        },
    ]

    all_scripts = []
    for alpha in alpha_vals:
        for s in settings:
            all_scripts.append(
                f"python3 ../main.py --algorithm fedvae --dataset mnist --num_users 10 --alpha "
                f"{alpha} --sample_ratio 0.5 --glob_epochs 5 --local_epochs 15 --should_log 1 --z_dim "
                f"10 --beta {s['beta']} --classifier_num_train_samples {s['classifier_train_samples']} "
                f"--decoder_num_train_samples {s['decoder_train_samples']} --classifier_epochs "
                f"{s['classifier_num_epochs']} --decoder_epochs {s['decoder_num_epochs']} --local_LR "
                f"{s['local_lr']} --decoder_LR {s['decoder_lr']} "
            )

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"hyperparam_runs_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
