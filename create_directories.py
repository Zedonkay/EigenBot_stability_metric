import os
import pandas as pd
import numpy as np
import shutil

def main():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()
    
    # Get unique values for terrains and control_types
    control_types = np.unique(data[:,0])
    terrains = {}
    legs = {}
    for control_type in control_types:
        terrains.update({control_type: np.unique(data[data[:,0] == control_type][:,1])})
    for control_type in control_types:
        legs.update({control_type: {}})
        for terrain in terrains.get(control_type):
            legs.get(control_type).update({terrain: np.unique(data[(data[:,0] == control_type) & (data[:,1] == terrain)][:,2])})

    # Remove the existing "3_results" directory if it exists
    if(os.path.exists("3_results")):
        shutil.rmtree("3_results")
    
    # Create the "3_results" directory
    os.mkdir("3_results")
    
    # Create directories for each control_type
    for control_type in control_types:
        os.mkdir(f"3_results/{control_type}")
        
        # Create directories for each terrain within each control_type
        for terrain in terrains.get(control_type):
            os.mkdir(f"3_results/{control_type}/{terrain}")
            
            # Create directories for each leg within each terrain
            for leg in legs.get(control_type).get(terrain):
                os.mkdir(f"3_results/{control_type}/{terrain}/leg{leg}")
    
    # Remove the existing "1_clean_data" directory if it exists
    if(os.path.exists("1_clean_data")):
        shutil.rmtree("1_clean_data")
    
    # Create the "1_clean_data" directory
    os.mkdir("1_clean_data")
    
    # Create directories for each control_type within "1_clean_data"
    for control_type in control_types:
        os.mkdir(f"1_clean_data/{control_type}")
        
        # Create directories for each terrain within each control_type
        for terrain in terrains.get(control_type):
            os.mkdir(f"1_clean_data/{control_type}/{terrain}")

if __name__ == "__main__":
    main()
