import os

import pandas as pd

from experiments.utils import csvs_to_dfs, to_csv

if __name__ == "__main__":
    # Convert ablation results to csvs
    experiment_dir = "../runs"
    if os.listdir(experiment_dir + "/csv") == None:
        to_csv(experiment_dir)

    # Convert csvs to dataframes
    dfs = csvs_to_dfs(experiment_dir + "/csv")

    for df_key in dfs.keys():
        print(df_key)
        print(dfs[df_key])
