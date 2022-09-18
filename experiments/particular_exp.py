import numpy as np

from utils import create_shell_files, populate_default_scripts

if __name__ == "__main__":
    hosts = np.array([19, 20, 21])
    all_scripts = [
        "python3 ../main.py --algorithm central_cvae --dataset mnist --sample_ratio 0.5 --glob_epochs 50 --z_dim 10 --beta 1.0 --local_LR 0.001 --should_log 1",
        "python3 ../main.py --algorithm central_cvae --dataset fashion --sample_ratio 0.5 --glob_epochs 50 --z_dim 100 --beta 1.0 --local_LR 0.001 --should_log 1",
        "python3 ../main.py --algorithm central_cvae --dataset svhn --sample_ratio 1.0 --glob_epochs 100 --z_dim 10 --beta 1.0 --local_LR 0.001 --should_log 1",
    ]

    print("Number of experiments:", len(all_scripts))

    create_shell_files(all_scripts, hosts, "particular_exp")
