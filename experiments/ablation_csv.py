import os
from pathlib import Path

from experiments.utils import csvs_to_dfs, to_csv

if __name__ == "__main__":
    # Convert ablation results to csvs
    experiment_dir = "../runs"
    out_path = Path(experiment_dir + "/csv/ablation.csv")

    if not os.path.exists(experiment_dir + "/csv"):
        to_csv(experiment_dir)

    # Convert csvs to dataframes
    dfs = csvs_to_dfs(experiment_dir + "/csv")

    for df_key in dfs.keys():
        print(df_key)
        print(dfs.get(df_key))
