import pandas as pd
import sys

def step_1_datapreprocessing(data_dir):
    """
    Load and merge time-series sensor data with demographic data.
    
    Parameters:
    -----------
    data_dir : str
        Directory path containing the CSV files
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe containing sensor data and demographic information
        
    Raises:
    -------
    SystemExit
        If required CSV files are not found in the specified directory
    """
    
    print("📊 Loading datasets...")
    
    try:
        # Load the main time-series sensor data
        train_df = pd.read_csv(f"{data_dir}train.csv")
        print("    ✅ Train data loaded")
        
        # Load the demographic data for each subject
        train_demographics_df = pd.read_csv(f"{data_dir}train_demographics.csv")
        print("    ✅ Demographics data loaded")
        
        print("\n🎉 Datasets loaded successfully!")
        print(f"    📈 Train data shape: {train_df.shape}")
        print(f"    👥 Demographics data shape: {train_demographics_df.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please make sure the CSV files ('train.csv', 'train_demographics.csv') are in the correct directory.")
        # Exit gracefully if files are not found
        sys.exit(1)
    
    # --- Merge DataFrames ---
    print("\n🔗 Merging dataframes on 'subject' column...")
    
    # We use a 'left' merge to ensure all sensor readings from train_df are kept.
    # The corresponding demographic information is added to each row based on the 'subject'.
    merged_df = pd.merge(train_df, train_demographics_df, on='subject', how='left')
    
    print("    ✅ Merge complete!")
    print(f"    📋 Merged dataframe shape: {merged_df.shape}")
    
    # --- Display Results ---
    print("\n🔍 Showing the first 5 rows of the merged dataframe:")
    print(merged_df.head())
    
    print("\n🏗️ Verifying the new columns have been added:")
    # List the last 10 columns to see the newly merged demographic features
    print(merged_df.columns[-10:])
    
    print("\n🔎 Checking for any missing values introduced by the merge (should be 0):")
    # This checks if any subject in train_df was not present in train_demographics_df
    missing_values_after_merge = merged_df['age'].isnull().sum()
    
    if missing_values_after_merge == 0:
        print(f"    ✅ Perfect! No missing demographic info after merge: {missing_values_after_merge}")
    else:
        print(f"    ⚠️ Warning! Rows with missing demographic info after merge: {missing_values_after_merge}")
    
    print("\n🎊 Data preprocessing completed successfully!")
    return merged_df





# Example usage:
if __name__ == "__main__":
    # Define your data directory path
    DATA_DIR = './'  # Update this path as needed
    
    # Call the preprocessing function
    processed_data = step_1_datapreprocessing(DATA_DIR)
    
    # The merged dataframe is now available as 'processed_data'
    print(f"\n🚀 Final processed data shape: {processed_data.shape}")