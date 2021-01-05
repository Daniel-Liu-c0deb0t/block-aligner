import os
import sys

import pandas as pd

base_remote_path = "ftp://ftp.ncbi.nlm.nih.gov/blast/matrices/"
base_local_path = "../matrices/"
names = sys.argv[1:] # BLOSUM62, etc.

os.makedirs(base_local_path, exist_ok = True)

for name in names:
    path = base_remote_path + name
    df = pd.read_csv(path, delim_whitespace = True, comment = "#")
    df = df.drop(index = "*")
    df = df.drop(columns = "*")

    min_val = -128

    # note: '[' = 'A' + 26
    for i in range(27):
        c = chr(ord("A") + i)
        if not c in df.index:
            df.loc[c, :] = min_val

    for i in range(32):
        c = chr(ord("A") + i)
        if not c in df.columns:
            df.loc[:, c] = min_val

    # alphabetically sort the amino acids
    df = df.sort_index(axis = 0)
    df = df.sort_index(axis = 1)

    for col in df.columns:
        df[col] = df[col].astype(int)

    print(name)
    print(df)
    print()

    df.to_csv(base_local_path + name, index = False, header = False)
