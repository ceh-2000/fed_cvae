import os

from experiments.utils import csvs_to_dfs, to_csv

if __name__ == "__main__":
    # Convert ablation results to csvs
    experiment_dir = "../runs"
    if not os.path.exists(experiment_dir + "/csv"):
        to_csv(experiment_dir)

    # Convert csvs to dataframes
    dfs = csvs_to_dfs(experiment_dir + "/csv")

    for df_key in dfs.keys():
        print(df_key)

        df = dfs[df_key]
        df.index.name = "run_name"

        df["accuracies"] = df[1]
        df = df.drop(1, axis=1)
        df["filenames"] = df.index
        df.index = range(0, len(df))

        seed_df = df["filenames"].str.extract(r"seed=(?P<seed>.*)")
        df["seed"] = seed_df

        df["exps"] = df["filenames"].str.extract(r"(should.*)_seed")
        df = df.drop("filenames", axis=1)

        df = df.groupby(["exps"]).agg({"accuracies": ["mean", "min", "max", "std"]})
        print(df)
