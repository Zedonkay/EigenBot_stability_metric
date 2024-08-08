import pandas as pd
import numpy as np
import filename_generation as fg
import state_space as ss

# Function to find the start and end indices of data based on a tolerance
def find_start_and_end(data, tolerance):
    start = find_start(data, tolerance)
    end = find_end(data, tolerance)
    return start, end

# Function to find the start index of data based on a tolerance
def find_start(data, tolerance):
    for i in range(len(data)-1):
        if np.linalg.norm(data[i] - data[i+1]) > tolerance:
            return i
    return 0

# Function to find the end index of data based on a tolerance
def find_end(data, tolerance):
    for i in range(len(data)-1, 0, -1):
        if np.linalg.norm(data[i] - data[i-1]) > tolerance:
            return i

# Function to truncate the data based on start and end indices
def retruncate(terrain,object,test, start, end):
    if not (start==0 and end==9999):
        # Generate the filename based on the input parameters
        filename = fg.filename_clean(terrain,object,test)
        
        # Read the raw data from the CSV file
        raw_test = pd.read_csv(filename)
        
        # Truncate the data based on the start and end indices
        if end != 9999:
            raw_test = raw_test.iloc[start:end]
        else:
            raw_test = raw_test.iloc[start:]
        
        # Save the truncated data back to the CSV file
        raw_test.to_csv(filename, index=False)

# Function to calculate angular velocities between quaternions
def angular_velocities(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

# Function to calculate angular velocities for a list of quaternions and times
def calculate_angular_velocities(quaternions, times):
    a_velocities = []
    for i in range(1, len(quaternions)):
        q1 = quaternions[i-1]
        q2 = quaternions[i]
        a_velocities.append(angular_velocities(q1, q2, times[i] - times[i-1]))
    return np.array(a_velocities)

def remove_duplicates(data):
    for i in range(len(data)-1,1,-1):
        if data[i][1]==data[i-1][1]:
            data = np.delete(data, i, axis=0)
    return data
# Main function
def main(terrain,object,test, tolerance):
    # Generate the filename for the raw data
    filename = fg.filename_raw_test(terrain,object,test)
    
    # Read the raw data from the CSV file
    df = pd.read_csv(filename)

    #remove duplicates
    data =remove_duplicates(df.values)

    df=pd.DataFrame(data,columns=df.columns)
    
    # Extract the 'pz' column as the data
    data = df[['pz']].values
    
    # Uncomment the following lines to window the data according to the tolerance
    # start, end = find_start_and_end(data, tolerance)
    # df = df[start:end]
    
    # Subtract the first timestamp from all timestamps
    df['timestamp'] = df['timestamp'] - df.iloc[0, 0]
    
    # Extract the quaternion and timestamp columns
    quaternion = df[['ow', 'ox', 'oy', 'oz']].values
    timestamp = df['timestamp'].values
    
    # Calculate the angular velocities between quaternions
    angular_velocities = calculate_angular_velocities(quaternion, timestamp)
    angular_velocities = np.insert(angular_velocities, 0, [0, 0, 0], axis=0)
    
    # Calculate the velocity, acceleration, jerk, and angular acceleration
    df['vx'] = np.gradient(df['px'], df['timestamp'])
    df['vy'] = np.gradient(df['py'], df['timestamp'])
    df['vz'] = np.gradient(df['pz'], df['timestamp'])
    
    df['ax'] = np.gradient(df['vx'], df['timestamp'])
    df['ay'] = np.gradient(df['vy'], df['timestamp'])
    df['az'] = np.gradient(df['vz'], df['timestamp'])
    
    df['jx'] = np.gradient(df['ax'], df['timestamp'])
    df['jy'] = np.gradient(df['ay'], df['timestamp'])
    df['jz'] = np.gradient(df['az'], df['timestamp'])
    
    df['wx'] = angular_velocities[:, 0]
    df['wy'] = angular_velocities[:, 1]
    df['wz'] = angular_velocities[:, 2]
    
    df['aa_x'] = np.gradient(df['wx'])
    df['aa_y'] = np.gradient(df['wy'])
    df['aa_z'] = np.gradient(df['wz'])
    
    quaternion = df.iloc[:, 4:8].values
    roll,pitch,yaw = ss.quaternion_to_euler(quaternion)
    df['roll'] = roll
    df['pitch'] = pitch
    df['yaw'] = yaw
    # Save the processed data to a new CSV file
    df.to_csv(fg.filename_clean(terrain,object,test), index=False)
