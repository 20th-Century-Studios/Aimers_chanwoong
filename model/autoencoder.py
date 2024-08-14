import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Autoencoder 모델 정의
def create_autoencoder(input_dim):
    encoding_dim = input_dim // 2
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    
    # Decoder
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    
    # Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder

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
    
    # Create the Autoencoder model
    autoencoder = create_autoencoder(input_dim=X_train.shape[1])
    
    # Train the Autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_valid, X_valid), verbose=1)
    
    # Predict the validation set (reconstruction)
    X_valid_pred = autoencoder.predict(X_valid)
    
    # Calculate reconstruction error
    mse = np.mean(np.power(X_valid - X_valid_pred, 2), axis=1)
    
    # Determine threshold for anomaly detection
    threshold = np.percentile(mse, 95)  # Use 95th percentile of reconstruction error as threshold
    
    # Classify as anomaly or normal
    y_pred_valid = np.where(mse > threshold, 0, 1)
    
    # Calculate F1 Score
    f1 = f1_score(y_valid, y_pred_valid, average='weighted')
    print(f'F1 Score for Fold {fold+1}: {f1}')
    print(f'Classification Report:\n {classification_report(y_valid, y_pred_valid)}')
    print(f'========================================================================================\n\n')

# Predict on the test set
X_test_pred = autoencoder.predict(X_test_scaled)

# Calculate reconstruction error for test data
mse_test = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

# Classify as anomaly or normal based on the same threshold
test_predictions = np.where(mse_test > threshold, 0, 1)
submission_list = ['Normal' if x == 1 else 'AbNormal' for x in test_predictions]

submission = pd.read_csv("../../submission.csv")
submission['target'] = submission_list

submission.to_csv('../output/submission_autoencoder.csv', index=False)
print("Test predictions saved to submission.csv")