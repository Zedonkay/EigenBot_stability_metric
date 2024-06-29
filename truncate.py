import pandas as pd
import numpy as np
import filename_generation as fg
from scipy.interpolate import interp1d

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

def compute_velocities(pos_x, pos_y, pos_z, timestamps):
    # Compute time differentials
    time_diffs = np.diff(timestamps)

    # Compute position differentials
    pos_x_diffs = np.diff(pos_x)
    pos_y_diffs = np.diff(pos_y)
    pos_z_diffs = np.diff(pos_z)

    # Interpolate velocities at timestamps where position data is available
    interp_func_x = interp1d(timestamps[:-1], pos_x_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(timestamps[:-1], pos_y_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_z = interp1d(timestamps[:-1], pos_z_diffs / time_diffs, kind='linear', fill_value='extrapolate')

    # Compute velocities at original timestamps
    vel_x = interp_func_x(timestamps)
    vel_y = interp_func_y(timestamps)
    vel_z = interp_func_z(timestamps)

    return vel_x, vel_y, vel_z     
  
def main(frequency,test,control_type,tolerance):
    filename = fg.filename_raw_test(frequency,test,control_type)
    df = pd.read_csv(filename)
    data = df[['pz']].values
    start,end = find_start_and_end(data,tolerance)
    df = df[start:end]
    vel_x, vel_y, vel_z = compute_velocities(df['px'].values,df['py'].values,df['pz'].values,df['timestamp'].values)
    df = df.assign(vel_x=vel_x,vel_y=vel_y,vel_z=vel_z)
    df.to_csv(fg.filename_clean(frequency,test,control_type),index=False)


if __name__ == '__main__':
    centralised = [[100,'1'],[140,'1'],[180,'1'],
                   [180,'2'],[180,'3'],[220,'1'],[220,'2'],
                   [220,'3'],[260,'1'],[260,'2'],[260,'3'],
                  [320,'1'],[320,'2'],
                   [330,'1'],[350,'1']]
   
    distributed =[[100,'1'],[140,'1'],[140,'2'],
                  [180,'1'],[180,'2'],[220,'1'],[220,'2'],
                  [260,'1'],[260,'2'],[260,'3'],[300,'1'],[300,'2'],[300,'3'],
                  [350,'1'],[350,'2'],[370,'1'],[370,'2'],[400,'1'],
                  [400,'2'],[500,'1'],[600,'1'],[700,'1'],[800,'1'],
                  [1000,'1'],[1040,'1'],[1120,'1'],[1160,'1'],[1200,'1'],
                  [1440,'1'],[1600,'1'], [1800,'1'],[2000,'1']]
    
    tolerance = 0.001
    print("running centralised")
    for i in centralised:
        main(i[0],i[1],'centralised',tolerance)
    print("running distributed")
    for i in distributed:
        main(i[0],i[1],'distributed',tolerance)
    print("done")