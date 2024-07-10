import os
import pandas as pd
import filename_generation as fg
import shutil

if __name__ =="__main__":
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()
    shutil.rmtree("6_Results/clean_data/centralised")
    os.mkdir("6_Results/clean_data/centralised")
    shutil.rmtree("6_Results/clean_data/distributed")
    os.mkdir("6_Results/clean_data/distributed")
    shutil.rmtree("1_clean_data")
    os.mkdir("1_clean_data")
    os.mkdir("1_clean_data/centralised")
    os.mkdir("1_clean_data/distributed")
    for file in data:
        print(f"{file[0]} {file[1]}hz test {file[2]}")
        if(not os.path.exists(fg.clean_create_folder_store_frequency(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder_store_frequency(file[1], file[2], file[0]))
        if(not os.path.exists(fg.clean_create_folder_store(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder_store(file[1], file[2], file[0]))
        if(not os.path.exists(fg.clean_create_folder(file[1], file[2], file[0]))):
            os.mkdir(fg.clean_create_folder(file[1], file[2], file[0]))
            
        