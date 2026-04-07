import os
import glob
import pandas as pd

def preprocess_data(base_dir="./runs", output_dir="./processed"):
    print("Starting data preprocessing...")
    
    labels_data = []
    sensor_data_dict = {} # dictionary to hold the master dataframes
    
    # find all folders in the base directory
    folders = glob.glob(os.path.join(base_dir, "*OpenWearable_Recording*"))
    
    if not folders:
        print(f"No folders found in {base_dir}")
        return

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        
        # 1. Extract Ride_ID and label from the folder name
        parts = folder_name.rsplit('_', 1) # label is rightmost number after underscore
        ride_id = parts[0] # left side of the last underscore
        try:
            label = float(parts[1]) # distraction score
        except ValueError:
            continue # skip if the end of the folder name isn't a valid number
            
        # save the label for this ride
        labels_data.append({"Ride_ID": ride_id, "Distraction_Score": label})
        
        # process every CSV file inside this specific folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)
            # skip raw accelerometer_1 data
            if "ACCELEROMETER_1" in file_name:
                continue

            # split after the first underscore to make optical temperature and temperature distinct
            sensor_name_part = os.path.splitext(file_name)[0].split('_', 1)[1]

            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
            
            # add the ride_ID so tsfresh knows which ride this belongs to
            df.insert(0, 'Ride_ID', ride_id)
            
            # dynamically rename columns (ignoring Ride_ID and timestamp)
            new_cols = {}
            for col in df.columns:
                if col not in ['Ride_ID', 'timestamp']:
                    # e.g. 'X' becomes 'ACCELEROMETER_X'
                    new_cols[col] = f"{sensor_name_part}_{col}"
            df.rename(columns=new_cols, inplace=True)

            # append to master dictionary
            if sensor_name_part not in sensor_data_dict:
                sensor_data_dict[sensor_name_part] = []
            sensor_data_dict[sensor_name_part].append(df)


    os.makedirs(output_dir, exist_ok=True)

    # save the labels file
    if labels_data:
        labels_df = pd.DataFrame(labels_data)
        labels_file = os.path.join(output_dir, "y_labels.csv")
        labels_df.to_csv(labels_file, index=False)
        print(f"Saved Labels: {labels_file} ({len(labels_df)} rides labeled)")

    # save the master CSV for each sensor
    for sensor_name, df_list in sensor_data_dict.items():
        master_df = pd.concat(df_list, ignore_index=True)
        output_file = os.path.join(output_dir, f"master_{sensor_name.lower()}.csv")
        master_df.to_csv(output_file, index=False)
        print(f"Saved Sensor Data: {output_file} ({len(master_df)} total rows across all rides)")

    print("\nPreprocessing complete, data is ready for tsfresh feature extraction")

if __name__ == "__main__":
    preprocess_data()