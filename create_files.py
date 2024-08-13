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
            elapsed_times = []
            previous_timestamp = None
            
            for data in leg_data:
                if data[1] == 1:
                    if previous_timestamp is not None:
                        elapsed_time = data[0] - previous_timestamp
                        if(elapsed_time>0 and elapsed_time<0.1):
                            elapsed_times.append(elapsed_time)
                    previous_timestamp = data[0]
            elapsed_times = np.array(elapsed_times)
            data = pd.DataFrame(elapsed_times, columns=['elapsed_time'])
            data.to_csv(fg.filename_clean(file[0], file[1], leg), index=False)
if __name__=="__main__":
   make_leg_files()