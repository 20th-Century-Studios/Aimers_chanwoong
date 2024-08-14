import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# Parameters for Local Outlier Factor
param = {
    "n_neighbors": 20,  # Number of neighbors to use for the LOF calculation
    "contamination": 0.1,  # The amount of contamination (outliers) in the data
    "novelty": True  # Enables LOF to be used for novelty detection
}

# Initialize the LOF model
model = LocalOutlierFactor(**param)

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

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_data)

# Generate KFold indices
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X_scaled))

# Loop over each fold for cross-validation
for fold in range(5):
    print(f'==================================== Fold {fold+1} ============================================')
    train_idx, valid_idx = folds[fold]
    
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Fit the LOF model on the training data
    model.fit(X_train)
    
    # Predict on the validation set (-1 for outliers, 1 for inliers)
    y_pred_valid = model.predict(X_valid)
    
    # Map LOF output to binary class labels
    y_pred_valid = np.where(y_pred_valid == -1, 0, 1)
    
    # Calculate F1 Score
    f1 = f1_score(y_valid, y_pred_valid, average='weighted')
    print(f'F1 Score for Fold {fold+1}: {f1}')
    print(f'Classification Report:\n {classification_report(y_valid, y_pred_valid)}')
    print(f'========================================================================================\n\n')

# Predict on the test set
test_predictions = model.predict(X_test_scaled)

# Map LOF output to binary class labels for test data
test_predictions = np.where(test_predictions == -1, 0, 1)
submission_list = ['Normal' if x == 1 else 'AbNormal' for x in test_predictions]

# If you want to save the predictions to a CSV file
output = pd.DataFrame({
    'Set ID': set_ID,
    'target': submission_list
})

output.to_csv('../output/submission_LOF.csv', index=False)
print("Test predictions saved to submission.csv")
