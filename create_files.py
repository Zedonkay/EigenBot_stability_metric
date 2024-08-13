import pandas as pd
import numpy as np
import filename_generation as fg

def make_filtered_csv():
    data = pd.read_csv('2_raw_data/running_info.csv')
    filtered_data = data.drop_duplicates(subset=['control_type', 'Terrain'])
    filtered_data.to_csv('2_raw_data/filtered_running_info.csv', index=False)
def make_leg_files():
    data = pd.read_csv('2_raw_data/filtered_running_info.csv')
    files = data.to_numpy()
    for file in files:
        print(f"Processing data for {file[0]} control on {file[1]} terrain")
        filename = fg.filename_raw_test(file[1], file[0], file[2])
        df = pd.read_csv(filename)
        for leg in range(1, 7):
            print(f"Processing leg {leg}")
            leg_data = df[['Timestamp',f'Contact_{leg}']]
            leg_data=leg_data.values
            stance_times = []
            swing_times = []
            phase = leg_data[0][1]
            stance_start=None
            swing_start=None
            stance_times = np.array(stance_times)
            swing_times = np.array(swing_times)
            for data in leg_data:
                if (stance_start==None and data[1]==1):
                    stance_start=data[0]
                if (swing_start==None and data[1]==0):
                    swing_start=data[0]
                if data[1] != phase:
                    if phase == 1:
                        stance_times = np.append(stance_times, data[0] - stance_start)
                        stance_start =data[0]
                    else:
                        swing_times = np.append(swing_times, data[0] - swing_start)
                        swing_start = data[0]
                    phase = data[1]
            while len(stance_times) < len(swing_times):
                stance_times = np.append(stance_times,0)
            while len(swing_times) < len(stance_times):
                swing_times = np.append(swing_times, 0)
            
            df_leg = pd.DataFrame({'stance': stance_times, 'swing': swing_times})
            # Save the dataframe to a csv file
            df_leg.to_csv(fg.filename_clean(file[0], file[1], leg), index=False)
if __name__=="__main__":
   make_leg_files()