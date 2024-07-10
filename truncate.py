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



def main(frequency,test,control_type,tolerance):
    filename = fg.filename_raw_test(frequency,test,control_type)
    df = pd.read_csv(filename)
    data = df[['pz']].values
    start,end = find_start_and_end(data,tolerance)
    df = df[start:end]
    df['timestamp']=df['timestamp']-df.iloc[0,0]
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

    df.to_csv(fg.filename_clean(frequency,test,control_type),index=False)
