import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Parameters for SVM Classifier
param = {
    "C": 1.0,  # Regularization parameter
    "kernel": 'rbf',  # Can be 'linear', 'poly', 'rbf', 'sigmoid', etc.
    "gamma": 'scale',  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    "random_state": 42
}

# Initialize the SVM Classifier
model = SVC(**param)

# Get the root directory and file names from the user
ROOT_DIR = "../../data"
train_file = "preprocessed_ml_train.csv"
test_file = "preprocessed_ml_test.csv"

# Construct the full paths
train_path = os.path.join(ROOT_DIR, train_file)
test_path = os.path.join(ROOT_DIR, test_file)

# Check if the files exist
if not os.path.isfile(train_path):
    print("Training file not found. Please check the filename and try again.")
    exit()

if not os.path.isfile(test_path):
    print("Test file not found. Please check the filename and try again.")
    exit()

# Load the datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
set_ID = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))['Set ID']

# Ensure the target column is present
target = 'target'  # Update this to your actual target column name
if target not in train_data.columns:
    print("Target column not found in the training DataFrame. Please check the target column name.")
    exit()

# Split the training data into features and target
X = train_data.drop(columns=[target])
y = train_data[target]

# Generate KFold indices
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X))

# Loop over each fold for cross-validation
for fold in range(5):
    print(f'==================================== Fold {fold+1} ============================================')
    train_idx, valid_idx = folds[fold]
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Fit the model with the resampled data
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the validation set
    y_pred = model.predict(X_valid)
    
    # Calculate F1 Score
    f1 = f1_score(y_valid, y_pred, average='weighted')  # Use 'macro' or 'micro' for different averaging methods
    print(f'F1 Score for Fold {fold+1}: {f1}')
    print(f'========================================================================================\n\n')

# Predict on the test set
X_test = test_data
test_predictions = model.predict(X_test)
submission_list = ['Normal' if x == 1 else 'AbNormal' for x in test_predictions]

# If you want to save the predictions to a CSV file
output = pd.DataFrame({
    'Set ID': set_ID,
    'prediction': test_predictions
})

submission = pd.read_csv("../../submission.csv")
submission['target'] = submission_list

output.to_csv('submission.csv', index=False)
print("Test predictions saved to submission.csv")
