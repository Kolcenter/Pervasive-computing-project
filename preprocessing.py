import os
import glob
import pandas as pd

def preprocess_data(base_dir="./", output_dir="./processed"):
    print("Starting data preprocessing...")
    
    labels_data = []
    sensor_data_dict = {} # Dictionary to hold our master dataframes
    
    # Find all folders in the base directory
    folders = glob.glob(os.path.join(base_dir, "*OpenWearable_Recording*"))
    
    if not folders:
        print(f"No folders found in {base_dir}")
        return

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        
        # 1. Extract Ride_ID and Label from the folder name
        # Example: "OpenWearable_Recording_2026-03-26T101341.966067_4"
        parts = folder_name.rsplit('_', 1) # Label is rightmost digit after underscore
        ride_id = parts[0] # Left side of the last underscore
        try:
            label = int(parts[1]) # distraction score 0-4
        except ValueError:
            continue # skip if the end of the folder name isn't a valid number
            
        # Save the label for this ride
        labels_data.append({"Ride_ID": ride_id, "Distraction_Score": label})
        
        # 2. Process every CSV file inside this specific folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            # skip feature
            if "ACCELEROMETER_1" in file_name:
                continue

            # Use split once after the first underscore and strip the file extension so multi-word sensors
            # like OPTICAL_TEMPERATURE_SENSOR and TEMPERATURE_SENSOR remain distinct.
            sensor_name_part = os.path.splitext(file_name)[0].split('_', 1)[1]
            # Read the CSV
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
            
            # Add the Ride_ID so tsfresh knows which ride this belongs to
            df.insert(0, 'Ride_ID', ride_id)
            
            # 3. Dynamically rename columns (ignoring Ride_ID and timestamp)
            # handles varying column structure
            new_cols = {}
            for col in df.columns:
                if col not in ['Ride_ID', 'timestamp']:
                    # E.g., 'X' becomes 'ACCELEROMETER_X'
                    new_cols[col] = f"{sensor_name_part}_{col}"
            df.rename(columns=new_cols, inplace=True)
            
            # 4. Append to our master dictionary
            if sensor_name_part not in sensor_data_dict:
                sensor_data_dict[sensor_name_part] = []
            sensor_data_dict[sensor_name_part].append(df)


    os.makedirs(output_dir, exist_ok=True)

    # Save the labels file
    if labels_data:
        labels_df = pd.DataFrame(labels_data)
        labels_file = os.path.join(output_dir, "y_labels.csv")
        labels_df.to_csv(labels_file, index=False)
        print(f"Saved Labels: {labels_file} ({len(labels_df)} rides labeled)")

    # Save the master CSV for each sensor
    for sensor_name, df_list in sensor_data_dict.items():
        master_df = pd.concat(df_list, ignore_index=True)
        output_file = os.path.join(output_dir, f"master_{sensor_name.lower()}.csv")
        master_df.to_csv(output_file, index=False)
        print(f"Saved Sensor Data: {output_file} ({len(master_df)} total rows across all rides)")

    print("\nPreprocessing complete, data is ready for tsfresh.")

if __name__ == "__main__":
    preprocess_data()