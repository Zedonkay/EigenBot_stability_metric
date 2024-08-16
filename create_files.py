import pandas as pd
import numpy as np
import filename_generation as fg

# Function to create a filtered CSV file
def make_filtered_csv():
    # Read the original CSV file
    data = pd.read_csv('2_raw_data/running_info.csv')
    
    # Drop duplicate rows based on 'control_type' and 'Terrain' columns
    filtered_data = data.drop_duplicates(subset=['control_type', 'Terrain'])
    
    # Save the filtered data to a new CSV file
    filtered_data.to_csv('2_raw_data/filtered_running_info.csv', index=False)

# Function to create leg files
def make_leg_files():
    # Read the filtered CSV file
    data = pd.read_csv('2_raw_data/filtered_running_info.csv')
    
    # Convert the data to a numpy array
    files = data.to_numpy()
    
    # Process each file in the array
    for file in files:
        # Print the control type and terrain for the current file
        print(f"Processing data for {file[0]} control on {file[1]} terrain")
        
        # Generate the filename for the current file
        filename = fg.filename_raw_test(file[1], file[0], file[2])
        
        # Read the data from the generated filename
        df = pd.read_csv(filename)
        
        # Process each leg (1 to 6)
        for leg in range(1, 7):
            # Print the current leg number
            print(f"Processing leg {leg}")
            
            # Extract the timestamp and contact data for the current leg
            leg_data = df[['Timestamp', f'Contact_{leg}']]
            leg_data = leg_data.values
            
            # Initialize variables for stance and swing times
            stance_times = []
            swing_times = []
            phase = leg_data[0][1]
            stance_start = None
            swing_start = None
            
            # Convert stance_times and swing_times to numpy arrays
            stance_times = np.array(stance_times)
            swing_times = np.array(swing_times)
            
            # Process each data point in the leg_data array
            for data in leg_data:
                # Check for stance start
                if stance_start == None and data[1] == 1:
                    stance_start = data[0]
                
                # Check for swing start
                if swing_start == None and data[1] == 0:
                    swing_start = data[0]
                
                # Check for phase change
                if data[1] != phase:
                    # Calculate stance time
                    if phase == 1:
                        stance_times = np.append(stance_times, data[0] - stance_start)
                        stance_start = data[0]
                    # Calculate swing time
                    else:
                        swing_times = np.append(swing_times, data[0] - swing_start)
                        swing_start = data[0]
                    
                    # Update the phase
                    phase = data[1]
            
            # Adjust the lengths of stance_times and swing_times
            while len(stance_times) < len(swing_times):
                stance_times = np.append(stance_times, 0)
            while len(swing_times) < len(stance_times):
                swing_times = np.append(swing_times, 0)
            
            # Create a new dataframe with stance and swing times
            df_leg = pd.DataFrame({'stance': stance_times, 'swing': swing_times})
            
            # Save the dataframe to a CSV file
            df_leg.to_csv(fg.filename_clean(file[0], file[1], leg), index=False)

# Execute the make_leg_files function if this script is run directly
if __name__ == "__main__":
    make_leg_files()