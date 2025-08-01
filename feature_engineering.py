import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Load Merged Data ---
# This assumes you have the 'merged_df' from the previous step.
# For a standalone script, you would reload it.
# Let's re-run the merge step quickly to ensure we have the data.
print("Loading and merging data...")
try:
    train_df = pd.read_csv('./train.csv')
    train_demographics_df = pd.read_csv('./train_demographics.csv')
    merged_df = pd.merge(train_df, train_demographics_df, on='subject', how='left')
    print("Data ready for feature engineering.")
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'train_demographics.csv' are in the directory.")
    exit()

# --- Feature Engineering Configuration ---

# Define the columns we want to extract features from
IMU_COLS = [col for col in merged_df.columns if 'acc_' in col or 'rot_' in col]
THM_COLS = [col for col in merged_df.columns if 'thm_' in col]
TOF_COLS = [col for col in merged_df.columns if 'tof_' in col]

# Define the aggregation functions
# These will be applied to each column for each sequence
AGGREGATIONS = {
    'mean', 'std', 'min', 'max', 'median', 'skew',
    lambda x: np.quantile(x, 0.25), # 25th percentile
    lambda x: np.quantile(x, 0.75), # 75th percentile
    lambda x: np.max(x) - np.min(x) # Range
}

# Rename the lambda functions for clear column names
AGGREGATIONS_RENAME = {
    "<lambda_0>": "q25",
    "<lambda_1>": "q75",
    "<lambda_2>": "range"
}

# --- Feature Engineering Function ---

def create_features_for_sequence(group_df):
    """
    Extracts features from a single sequence's dataframe.
    A 'group' here is the dataframe for one unique sequence_id.
    """
    # Isolate the 'Gesture' phase for primary feature extraction
    gesture_df = group_df[group_df['behavior'] == 'Gesture']

    # --- Handle IMU-only case ---
    # Check if thermopile/ToF data is missing for this sequence
    # We check the first value of the first column for nulls
    is_imu_only = gesture_df[THM_COLS[0]].isnull().all()

    # --- IMU Features ---
    imu_features = gesture_df[IMU_COLS].agg(AGGREGATIONS).stack()

    # --- Thermopile (THM) Features ---
    if not is_imu_only:
        thm_features = gesture_df[THM_COLS].agg(AGGREGATIONS).stack()
    else:
        # Create placeholder features if data is missing
        # We create a multi-index series filled with 0s to match the structure
        idx = pd.MultiIndex.from_product([THM_COLS, list(AGGREGATIONS_RENAME.values()) + ['mean', 'std', 'min', 'max', 'median', 'skew']], names=['level_0', 'level_1'])
        thm_features = pd.Series(0, index=idx)


    # --- Time-of-Flight (TOF) Features ---
    if not is_imu_only:
        tof_features = gesture_df[TOF_COLS].agg(AGGREGATIONS).stack()
    else:
        # Create placeholder features if data is missing
        idx = pd.MultiIndex.from_product([TOF_COLS, list(AGGREGATIONS_RENAME.values()) + ['mean', 'std', 'min', 'max', 'median', 'skew']], names=['level_0', 'level_1'])
        tof_features = pd.Series(0, index=idx)

    # --- Combine all features for the sequence ---
    # The .unstack() and .stack() methods can be complex. Here we just combine the series.
    combined_features = pd.concat([imu_features, thm_features, tof_features])

    return combined_features

# --- Main Execution ---

print("\nStarting feature engineering process...")
print(f"Processing {merged_df['sequence_id'].nunique()} unique sequences.")

# Group by sequence_id and apply the feature engineering function
# tqdm adds a progress bar, which is very helpful for long processes
tqdm.pandas(desc="Creating Features")
features_df = merged_df.groupby('sequence_id').progress_apply(create_features_for_sequence)

# The result is a multi-index DataFrame. Let's flatten it.
features_df = features_df.unstack()
features_df.columns = ['_'.join(map(str, col)).strip() for col in features_df.columns.values]
features_df.rename(columns=lambda c: c.replace('<lambda_0>', 'q25').replace('<lambda_1>', 'q75').replace('<lambda_2>', 'range'), inplace=True)


# --- Add Target and Metadata ---
# Get the target 'gesture' and other static info for each sequence
# We can get this from the first row of each sequence group
metadata = merged_df.groupby('sequence_id').first()
features_df['gesture'] = metadata['gesture']
features_df['sequence_type'] = metadata['sequence_type']
features_df['subject'] = metadata['subject']

# Reset index to make 'sequence_id' a column
features_df.reset_index(inplace=True)

# --- Display Results ---
print("\nFeature engineering complete.")
print(f"Shape of the final features dataframe: {features_df.shape}")

print("\nShowing the first 5 rows of the new features dataframe:")
print(features_df.head())

print("\nExample of new feature columns:")
print(features_df.columns[10:20].tolist())

# Save the features to a file for the next step (model training)
features_df.to_csv('train_features.csv', index=False)
print("\nFeatures saved to 'train_features.csv'")
