import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Parameters for Isolation Forest
param = {
    "n_estimators": 100,  # Number of base estimators in the ensemble
    "max_samples": 'auto',  # Number of samples to draw from X to train each base estimator
    "contamination": 0.1,  # The amount of contamination (outliers) in the data
    "random_state": 42
}

# Initialize the Isolation Forest model
model = IsolationForest(**param)

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

# Encode the target column if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_data[target])

# Split the training data into features and target
X = train_data.drop(columns=[target])
y = y_encoded  # Isolation Forest expects the target to be binary, so encoding is used

# Generate KFold indices
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X))

# Loop over each fold for cross-validation
for fold in range(5):
    print(f'==================================== Fold {fold+1} ============================================')
    train_idx, valid_idx = folds[fold]
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    # Fit the Isolation Forest model
    model.fit(X_train)
    
    # Predict on the validation set (-1 for outliers, 1 for inliers)
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    
    # Map Isolation Forest output to binary class labels
    y_pred_valid = np.where(y_pred_valid == -1, 0, 1)
    
    # Calculate F1 Score
    f1 = f1_score(y_valid, y_pred_valid, average='weighted')
    print(f'F1 Score for Fold {fold+1}: {f1}')
    print(f'Classification Report:\n {classification_report(y_valid, y_pred_valid)}')
    print(f'========================================================================================\n\n')

# Predict on the test set
X_test = test_data
test_predictions = model.predict(X_test)

# Map Isolation Forest output to binary class labels for test data
test_predictions = np.where(test_predictions == -1, 0, 1)
submission_list = ['Normal' if x == 1 else 'AbNormal' for x in test_predictions]

submission = pd.read_csv("../../submission.csv")
submission['target'] = submission_list

submission.to_csv('../output/submission_isolation_forest.csv', index=False)
print("Test predictions saved to submission.csv")
