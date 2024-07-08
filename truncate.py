import pandas as pd
import numpy as np
import filename_generation as fg
from scipy.interpolate import interp1d
def compute_velocities(pos_x, pos_y, pos_z, timestamp):
    # Compute time differentials
    time_diffs = np.diff(timestamp)

    # Compute position differentials
    pos_x_diffs = np.diff(pos_x)
    pos_y_diffs = np.diff(pos_y)
    pos_z_diffs = np.diff(pos_z)

    # Interpolate velocities at timestamp where position data is available
    interp_func_x = interp1d(timestamp[:-1], pos_x_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(timestamp[:-1], pos_y_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_z = interp1d(timestamp[:-1], pos_z_diffs / time_diffs, kind='linear', fill_value='extrapolate')

    # Compute velocities at original timestamp
    vel_x = interp_func_x(timestamp)
    vel_y = interp_func_y(timestamp)
    vel_z = interp_func_z(timestamp)

    return vel_x, vel_y, vel_z
def compute_accelerations(vel_x, vel_y, vel_z, timestamp):
    # Compute time differentials
    time_diffs = np.diff(timestamp)

    # Compute velocity differentials
    vel_x_diffs = np.diff(vel_x)
    vel_y_diffs = np.diff(vel_y)
    vel_z_diffs = np.diff(vel_z)

    # Interpolate accelerations at timestamp where velocity data is available
    interp_func_x = interp1d(timestamp[:-1], vel_x_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(timestamp[:-1], vel_y_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_z = interp1d(timestamp[:-1], vel_z_diffs / time_diffs, kind='linear', fill_value='extrapolate')

    # Compute accelerations at original timestamp
    acc_x = interp_func_x(timestamp)
    acc_y = interp_func_y(timestamp)
    acc_z = interp_func_z(timestamp)

    return acc_x, acc_y, acc_z
def compute_jerks(acc_x, acc_y, acc_z, timestamp):
    # Compute time differentials
    time_diffs = np.diff(timestamp)

    # Compute acceleration differentials
    acc_x_diffs = np.diff(acc_x)
    acc_y_diffs = np.diff(acc_y)
    acc_z_diffs = np.diff(acc_z)

    # Interpolate jerks at timestamp where acceleration data is available
    interp_func_x = interp1d(timestamp[:-1], acc_x_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(timestamp[:-1], acc_y_diffs / time_diffs, kind='linear', fill_value='extrapolate')
    interp_func_z = interp1d(timestamp[:-1], acc_z_diffs / time_diffs, kind='linear', fill_value='extrapolate')

    # Compute jerks at original timestamp
    jerk_x = interp_func_x(timestamp)
    jerk_y = interp_func_y(timestamp)
    jerk_z = interp_func_z(timestamp)

    return jerk_x, jerk_y, jerk_z
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
