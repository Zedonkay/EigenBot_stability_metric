import os
import pandas as pd
import numpy as np
import shutil

def main():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()
    
    # Get unique values for objects and terrains
    terrains = np.unique(data[:,0])  # Get unique values from the first column
    objects = {}  # Dictionary to store objects for each terrain
    tests = {}  # Dictionary to store tests for each object within each terrain
    
    # Iterate over terrains and find unique objects for each terrain
    for terrain in terrains:
        objects.update({terrain: np.unique(data[data[:,0] == terrain][:,1])})
    
    # Iterate over terrains and objects to find unique tests for each object within each terrain
    for terrain in terrains:
        tests.update({terrain: {}})
        for object in objects.get(terrain):
            tests.get(terrain).update({object: np.unique(data[(data[:,0] == terrain) & (data[:,1] == object)][:,2])})

    # Remove the existing "3_results" directory if it exists
    if(os.path.exists("3_results")):
        shutil.rmtree("3_results")
    
    # Create the "3_results" directory
    os.mkdir("3_results")
    
    # Create directories for each terrain
    for terrain in terrains:
        os.mkdir(f"3_results/{terrain}")
        
        # Create directories for each object within each terrain
        for object in objects.get(terrain):
            os.mkdir(f"3_results/{terrain}/{object}")
            
            # Create directories for each test within each object
            for test in tests.get(terrain).get(object):
                os.mkdir(f"3_results/{terrain}/{object}/test{test}")
    
    # Remove the existing "1_clean_data" directory if it exists
    if(os.path.exists("1_clean_data")):
        shutil.rmtree("1_clean_data")
    
    # Create the "1_clean_data" directory
    os.mkdir("1_clean_data")
    
    # Create directories for each terrain within "1_clean_data"
    for terrain in terrains:
        os.mkdir(f"1_clean_data/{terrain}")
        
        # Create directories for each object within each terrain
        for object in objects.get(terrain):
            os.mkdir(f"1_clean_data/{terrain}/{object}")

if __name__ == "__main__":
    main()
