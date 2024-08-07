import os

import numpy as np
import pandas as pd

ROOT_DIR = "../../data"

# Data load
file_name = input("Please enter the CSV filename (e.g., train.csv): ")

file_path = os.path.join(ROOT_DIR, file_name)

if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
else:
    print("File not found. Please check the filename and try again.")


#### categorical columns preprocess ####

categorical_columns = df.select_dtypes(include=['object', 'category'])

categorical_columns['target'] = categorical_columns['target'].map({'AbNormal': 0, 'Normal': 1})

unique_counts = categorical_columns.nunique()
categorical_columns = categorical_columns[unique_counts[(unique_counts < 10) & (unique_counts > 1)].index]

categorical_columns = categorical_columns.drop(columns=['Model.Suffix - Dam', 'Model.Suffix - AutoClave', 'Model.Suffix - Fill1', 'Model.Suffix - Fill2'])
categorical_columns = categorical_columns.dropna(axis=1)

# One-Hot Encoding
categorical_columns = pd.get_dummies(categorical_columns, columns=['Equipment - Dam', 'Judge Value.3 - AutoClave', 'Equipment - Fill1', 'Equipment - Fill2']) # , 'Collect Result.17 - Fill2'


### Numeric columns preprocess ###

numeric_columns = df.select_dtypes(include=[np.number])


#### Combine ####

df_combined = pd.concat([numeric_columns, categorical_columns], axis=1)

cols = list(df_combined.columns)
cols.append(cols.pop(cols.index('target')))
df_combined = df_combined[cols]

##### Preprocessed data save ####
df_combined.to_csv(ROOT_DIR+ "preprocessed_ml.csv", index=False)



