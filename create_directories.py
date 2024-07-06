import os
import pandas as pd
import filename_generation as fg

if __name__ =="__main__":
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()
    for file in data:
        print(f"{file[0]} {file[1]}hz test {file[2]}")
        if(not os.path.exists(fg.clean_create_folder_store_frequency(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder_store_frequency(file[1], file[2], file[0]))
        if(not os.path.exists(fg.clean_create_folder_store(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder_store(file[1], file[2], file[0]))
        if(not os.path.exists(fg.clean_create_folder(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder(file[1], file[2], file[0]))
            
        