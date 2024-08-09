import os
import numpy as np
import pandas as pd

ROOT_DIR = "C:/Users/qh123/OneDrive/문서/Aimers/data"

# Function to preprocess the data
def preprocess_data(df, is_train=True):
    # Categorical columns preprocess
    categorical_columns = df.select_dtypes(include=['object', 'category'])
    if is_train:
        # Map target values if it's training data
        if 'target' in categorical_columns.columns:
            categorical_columns['target'] = categorical_columns['target'].map({'AbNormal': 0, 'Normal': 1})
        
    # Filter categorical columns based on unique values
    unique_counts = categorical_columns.nunique()
    categorical_columns = categorical_columns[unique_counts[(unique_counts < 10) & (unique_counts > 1)].index]
    
    # Drop specific columns
    drop_cols = ['Model.Suffix_Dam', 'Model.Suffix_AutoClave', 'Model.Suffix_Fill1', 'Model.Suffix_Fill2']
    categorical_columns = categorical_columns.drop(columns=drop_cols, errors='ignore')
    categorical_columns = categorical_columns.dropna(axis=1)
    
    # One-Hot Encoding
    categorical_columns = pd.get_dummies(categorical_columns, columns=['Equipment_Dam', 'Chamber Temp. Judge Value_AutoClave', 'Equipment_Fill1', 'Equipment_Fill2'])

    # Numeric columns preprocess
    numeric_columns = df.select_dtypes(include=[np.number])
    numeric_columns = numeric_columns.dropna(axis=1)
    
    # Combine numeric and categorical columns
    df_combined = pd.concat([numeric_columns, categorical_columns], axis=1)
    
    return df_combined

# Load and preprocess training data
train_file_name = input("Please enter the training CSV filename (e.g., train.csv): ")
train_file_path = os.path.join(ROOT_DIR, train_file_name)

if os.path.isfile(train_file_path):
    df_train = pd.read_csv(train_file_path)
    print("Training file loaded successfully.")
    print("Start preprocessing training data")
    df_train_processed = preprocess_data(df_train, is_train=True)
else:
    print("Training file not found. Please check the filename and try again.")
    exit()

# Save preprocessed training data
train_processed_file_name = "preprocessed_ml_" + train_file_name
train_processed_file_path = os.path.join(ROOT_DIR, train_processed_file_name)
df_train_processed.to_csv(train_processed_file_path, index=False)
print("Training data preprocessed and saved.")

# Load and preprocess test data
test_file_name = input("Please enter the test CSV filename (e.g., test.csv): ")
test_file_path = os.path.join(ROOT_DIR, test_file_name)

if os.path.isfile(test_file_path):
    df_test = pd.read_csv(test_file_path)
    print("Test file loaded successfully.")
    print("Start preprocessing test data")
    df_test_processed = preprocess_data(df_test, is_train=False)
else:
    print("Test file not found. Please check the filename and try again.")
    exit()

# Save preprocessed test data
test_processed_file_name = "preprocessed_ml_" + test_file_name
test_processed_file_path = os.path.join(ROOT_DIR, test_processed_file_name)
df_test_processed.to_csv(test_processed_file_path, index=False)
print("Test data preprocessed and saved.")
