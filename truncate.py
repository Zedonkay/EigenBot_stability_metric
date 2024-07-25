import pandas as pd
import numpy as np
import filename_generation as fg

def find_start_and_end(data,tolerance):
    start = find_start(data,tolerance)
    end = find_end(start,data,tolerance)
    return start,end
    
def find_start(data,tolerance):
    ret = []
    for i in range(len(data)-1):
        if np.linalg.norm(data[i]-data[i+1]>tolerance):
            return i
        else:
           if(i<500):
               ret.append(np.linalg.norm(data[i]-data[i+1]))
    return 0

        
def find_end(start,data,tolerance):
    for i in range(len(data)-1,0,-1):
        if np.linalg.norm(data[i]-data[i-1]>tolerance):
            return i
def clean_data(data):
    to_remove = []
    for i in range(1,len(data)-1):
        if(data[i-1][0]==data[i][0]):
            to_remove.append(i)
    return np.delete(data,to_remove,0)

        


def main(type,tolerance):
    filename = fg.filename_raw_data(type)
    df = pd.read_csv(filename)
    data = clean_data(df.values)
    df = pd.DataFrame(data,columns=df.columns)
    data = df['pz'].values
    start,end = find_start_and_end(data,tolerance)
    df = df[start:end]
    df['timestamp']=df['timestamp']-df.iloc[1,0]
    timestamp = df['timestamp'].values
    df['vx'] = np.gradient(df['px'], df['timestamp'])
    df['vy'] = np.gradient(df['py'], df['timestamp'])
    df['vz'] = np.gradient(df['pz'], df['timestamp'])
    
    df['ax'] = np.gradient(df['vx'], df['timestamp'])
    df['ay'] = np.gradient(df['vy'], df['timestamp'])
    df['az'] = np.gradient(df['vz'], df['timestamp'])

    df['jx'] = np.gradient(df['ax'], df['timestamp'])
    df['jy'] = np.gradient(df['ay'], df['timestamp'])
    df['jz'] = np.gradient(df['az'], df['timestamp'])

    df.to_csv(fg.filename_clean_data(type),index=False)

# Function to truncate the data based on start and end indices
def retruncate(type, start, end):
    # Generate the filename based on the input parameters
    filename = fg.filename_clean_data(type)
    
    # Read the raw data from the CSV file
    raw_test = pd.read_csv(filename)
    
    # Truncate the data based on the start and end indices
    if end != 9999:
        raw_test = raw_test.iloc[start:end]
    else:
        raw_test = raw_test.iloc[start:]
    
    # Save the truncated data back to the CSV file
    raw_test.to_csv(filename, index=False)