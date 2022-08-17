import numpy as np


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    hosts = np.array(
        [2, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    )
    all_scripts = []

    # By default we include weighting, averaging, and fine-tuning
    default_script = "python3 ../main.py --algorithm fedvae --dataset mnist --num_users 10 --alpha 0.01 --sample_ratio 0.5 --glob_epochs 1 --local_epochs 15 --z_dim 10 --beta 1.0 --classifier_num_train_samples 5000 --decoder_num_train_samples 5000 --classifier_epochs 10 --decoder_epochs 7 --local_LR 0.001 --decoder_LR 0.01 --should_log 1 --use_adam 1"

    all_seeds = [1588, 1693, 7089, 4488, 3776]
    all_exps = ["should_weight_exp", "should_avg_exp", "should_fine_tune_exp"]
    all_vals = [0, 1]

    for seed in all_seeds:
        for exp in all_exps:
            for val in all_vals:
                run_name = f"runs/fedvae_{exp}={val}_seed={seed}"
                all_scripts.append(
                    default_script
                    + f" --seed {seed} --{exp} {val} --cur_run_name {run_name}"
                )

    print("Number of experiments:", len(all_scripts))

    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"ablation_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1
