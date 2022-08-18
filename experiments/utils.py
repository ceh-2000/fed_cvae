import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def create_shell_files(all_scripts, hosts, file_name):
    counter = 0
    for i in split(all_scripts, len(hosts)):
        shell_file_name = f"{file_name}_{hosts[counter]}.sh"
        with open(shell_file_name, "w") as f:
            for c in i:
                f.write(c)
                f.write("\n")
        counter += 1


def tabulate_events(dpath):
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload()
        for dname in os.listdir(dpath)
    ]

    tags = summary_iterators[0].Tags()["scalars"]

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + ".csv"
    folder_path = os.path.join(dpath, "csv")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def csvs_to_dfs(dir):
    dfs = {}
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            df = pd.read_csv(f)
            dfs[filename.strip(".csv")] = df.drop(["Unnamed: 0"], axis=1).T

    return dfs
