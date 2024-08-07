import os

import numpy as np
import pandas as pd


# Data Load
def read_excel_file(file_path: str, header: int = None) -> pd.DataFrame:
    csv_file = file_path.replace(".xlsx", ".csv")

    if not os.path.exists(csv_file):
        print("Converting excel to csv...")
        if header:
            df = pd.read_excel(file_path, header=header)
        else:
            df = pd.read_excel(file_path)

        df.to_csv(csv_file, index=False)
        print(f"  {file_path} -> {csv_file}")
        return df
    else:
        print(f"  Reading {csv_file}")
        return pd.read_csv(csv_file, low_memory=False)
    
ROOT_DIR = "../../data/"
RANDOM_STATE = 110

X_Dam = read_excel_file(os.path.join(ROOT_DIR, "Dam dispensing.xlsx"), header=1)

X_AutoClave = read_excel_file(
    os.path.join(ROOT_DIR, "Auto clave.xlsx"), header=1
)

X_Fill1 = read_excel_file(
    os.path.join(ROOT_DIR, "Fill1 dispensing.xlsx"), header=1
)

X_Fill2 = read_excel_file(
    os.path.join(ROOT_DIR, "Fill2 dispensing.xlsx"), header=1
)

y = pd.read_csv(os.path.join(ROOT_DIR, "train_y.csv"))


# Merge Data
# Rename columns
X_Dam.columns = [i + " - Dam" for i in X_Dam.columns]
X_AutoClave.columns = [i + " - AutoClave" for i in X_AutoClave.columns]
X_Fill1.columns = [i + " - Fill1" for i in X_Fill1.columns]
X_Fill2.columns = [i + " - Fill2" for i in X_Fill2.columns]
X_Dam = X_Dam.rename(columns={"Set ID - Dam": "Set ID"})
X_AutoClave = X_AutoClave.rename(columns={"Set ID - AutoClave": "Set ID"})
X_Fill1 = X_Fill1.rename(columns={"Set ID - Fill1": "Set ID"})
X_Fill2 = X_Fill2.rename(columns={"Set ID - Fill2": "Set ID"})

# Merge X
X = pd.merge(X_Dam, X_AutoClave, on="Set ID")
X = pd.merge(X, X_Fill1, on="Set ID")
X = pd.merge(X, X_Fill2, on="Set ID")
X = X.drop(X[X.duplicated(subset="Set ID")].index).reset_index(drop=True)

# Merge X and y
df_merged = pd.merge(X, y, "inner", on="Set ID")

# Drop columns with more than half of the values missing
drop_cols = []
for column in df_merged.columns:
    if (df_merged[column].notnull().sum() // 2) < df_merged[
        column
    ].isnull().sum():
        drop_cols.append(column)
df_merged = df_merged.drop(drop_cols, axis=1)

# Drop Lot ID
df_merged = df_merged.drop("LOT ID - Dam", axis=1)
df_merged.to_csv(ROOT_DIR+ "merged.csv", index=False)