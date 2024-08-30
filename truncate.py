import pandas as pd
import numpy as np
import filename_generation as fg

def find_start_and_end(data, tolerance):
    """
    Find the start and end indices of data based on a tolerance value.

    Parameters:
    - data (numpy.ndarray): The input data.
    - tolerance (float): The tolerance value.

    Returns:
    - start (int): The start index.
    - end (int): The end index.
    """
    start = find_start(data, tolerance)
    end = find_end(start, data, tolerance)
    return start, end


def find_start(data, tolerance):
    """
    Find the start index of data based on a tolerance value.

    Parameters:
    - data (numpy.ndarray): The input data.
    - tolerance (float): The tolerance value.

    Returns:
    - start (int): The start index.
    """
    ret = []
    for i in range(len(data)-1):
        # Check if the norm of the difference between two consecutive data points is greater than the tolerance
        if np.linalg.norm(data[i] - data[i+1]) > tolerance:
            return i
        else:
            # If the norm is less than the tolerance, append it to the ret list
            if i < 500:
                ret.append(np.linalg.norm(data[i] - data[i+1]))
    return 0


def find_end(start, data, tolerance):
    """
    Find the end index of data based on a start index and a tolerance value.

    Parameters:
    - start (int): The start index.
    - data (numpy.ndarray): The input data.
    - tolerance (float): The tolerance value.

    Returns:
    - end (int): The end index.
    """
    for i in range(len(data)-1, 0, -1):
        # Check if the norm of the difference between two consecutive data points is greater than the tolerance
        if np.linalg.norm(data[i] - data[i-1]) > tolerance:
            return i

def angular_velocities(q1, q2, dt):
    """
    Calculate the angular velocities given two quaternions and the time difference.

    Parameters:
    - q1 (numpy.ndarray): The first quaternion.
    - q2 (numpy.ndarray): The second quaternion.
    - dt (float): The time difference.

    Returns:
    - velocities (numpy.ndarray): The calculated angular velocities.
    """
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def calculate_angular_velocities(quaternions, times):
    """
    Calculate the angular velocities from a list of quaternions and corresponding times.

    Parameters:
    - quaternions (numpy.ndarray): The input quaternions.
    - times (numpy.ndarray): The corresponding times.

    Returns:
    - a_velocities (numpy.ndarray): The calculated angular velocities.
    """
    a_velocities = []
    for i in range(0, len(quaternions)):
        if i==0:
            a_velocities.append(np.array([0, 0, 0]))
            continue
        q1 = quaternions[i-1]
        q2 = quaternions[i]
        a_velocities.append(angular_velocities(q1, q2, times[i] - times[i-1]))
    return np.array(a_velocities)

def calculate_euler_angles(quaternions):
    """
    Calculate the Euler angles from a list of quaternions.

    Parameters:
    - quaternions (numpy.ndarray): The input quaternions.

    Returns:
    - euler_angles (numpy.ndarray): The calculated Euler angles.
    """
    euler_angles = []
    for quaternion in quaternions:
        # Extract the components of the quaternion
        qw, qx, qy, qz = quaternion
        
        # Calculate the roll, pitch, and yaw angles
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(2 * (qw * qy - qx * qz))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        
        # Append the Euler angles to the list
        euler_angles.append([roll, pitch, yaw])
    
    return np.array(euler_angles)

def main(date, terrain, trial, tolerance):
    """
    Main function to process the data.

    Parameters:
    - date (str): The date of the data.
    - terrain (str): The terrain of the data.
    - trial (str): The trial of the data.
    - tolerance (float): The tolerance value.
    """
    # Generate the filename based on the input parameters
    filename = fg.filename_raw_data(date, terrain, trial)
    
    # Read the raw data from the CSV file
    df = pd.read_csv(filename)
    
    df['timestamp'] = df['stamp'] - df['stamp'][0]
    df['timestamp'] = df['timestamp'] / 1e9

    # Calculate angular velocities
    quaternions = df[['ow', 'ox', 'oy', 'oz']].values
    times = df['timestamp'].values
    a_velocities = calculate_angular_velocities(quaternions, times)
    
    df['wx'] = a_velocities[:, 0]
    df['wy'] = a_velocities[:, 1]
    df['wz'] = a_velocities[:, 2]

    # Calculate angular acceleration
    df['aa_x'] = np.gradient(df['wx'], df['timestamp'])
    df['aa_y'] = np.gradient(df['wy'], df['timestamp'])
    df['aa_z'] = np.gradient(df['wz'], df['timestamp'])

    # Calculate Euler angles
    euler_angles = calculate_euler_angles(quaternions)
    df['roll'] = euler_angles[:, 0]
    df['pitch'] = euler_angles[:, 1]
    df['yaw'] = euler_angles[:, 2]
    
    # Calculate the velocity components (vx, vy, vz) using gradient
    df['vx'] = np.gradient(df['px'], df['timestamp'])
    df['vy'] = np.gradient(df['py'], df['timestamp'])
    df['vz'] = np.gradient(df['pz'], df['timestamp'])
    
    # Calculate the acceleration components (ax, ay, az) using gradient
    df['ax'] = np.gradient(df['vx'], df['timestamp'])
    df['ay'] = np.gradient(df['vy'], df['timestamp'])
    df['az'] = np.gradient(df['vz'], df['timestamp'])
    
    # Calculate the jerk components (jx, jy, jz) using gradient
    df['jx'] = np.gradient(df['ax'], df['timestamp'])
    df['jy'] = np.gradient(df['ay'], df['timestamp'])
    df['jz'] = np.gradient(df['az'], df['timestamp'])
    
    #calculate forward velocity
    positions = df[['px', 'py', 'pz']].values
    forward_velocities = calculate_forward_velocity(positions,quaternions,1,times,date,terrain,trial)
    df['fw'] = forward_velocities

    forward_velocities2= calculate_forward_velocity(positions,quaternions,2,times,date,terrain,trial)
    df['fw2'] = forward_velocities2
    
    # Save the processed data to a new CSV file
    df.to_csv(fg.filename_clean_data(date, terrain, trial), index=False)

def calculate_forward_velocity(positions, quaternions,version,times,date,terrain,trial):
    """
    Calculate the forward velocity of the robot based on positions and quaternions.

    Parameters:
    - positions (numpy.ndarray): The positions of the robot.
    - quaternions (numpy.ndarray): The quaternions representing the orientation of the robot.

    Returns:
    - forward_velocities (numpy.ndarray): The calculated forward velocities.
    """
    forward_velocities = []
    forward_vectors=[]
    forward_vectors2=[]
    for i in range(len(positions)):
        if i == 0:
            forward_velocities.append(0)
            forward_vectors.append(np.array([0, 0, 0]))
            forward_vectors2.append(np.array([0, 0, 0]))
            continue
        # Get the current position and quaternion
        current_position = positions[i]
        current_quaternion = quaternions[i]
        
        # Get the previous position and quaternion
        previous_position = positions[i-1]
        previous_quaternion = quaternions[i-1]

        # Get the current time
        current_time = times[i]

        # Get the previous time
        previous_time = times[i-1]
        
        # Calculate the forward vector of the current orientation
        if version == 1:
            forward_vector = calculate_forward_vector(current_quaternion)
            forward_vectors.append(forward_vector)
        else:
            forward_vector = calculate_forward_vector2(current_quaternion)
            forward_vectors2.append(forward_vector)

        

        
        # Calculate the displacement vector between the current and previous positions
        displacement_vector = current_position - previous_position
        
        # Calculate the forward velocity by projecting the displacement vector onto the forward vector
        forward_displacement = np.dot(displacement_vector, forward_vector)

        #Calculate the forward velocity
        forward_velocity = forward_displacement / (current_time - previous_time)

        
        forward_velocities.append(forward_velocity)
    if version == 1:
        df = pd.DataFrame(np.column_stack([times, forward_vectors]), columns=['timestamp', 'fx', 'fy', 'fz'])
        df.to_csv(fg.filename_store_data(date,terrain,trial)+'forward_vectors.csv', index=False)
    else:
        df = pd.DataFrame(np.column_stack([times, forward_vectors2]), columns=['timestamp', 'fx', 'fy', 'fz'])
        df.to_csv(fg.filename_store_data(date,terrain,trial)+'forward_vectors2.csv', index=False)
    
    return np.array(forward_velocities)

def calculate_forward_vector(quaternion):
    """
    Calculate the forward vector based on a quaternion representing the orientation.

    Parameters:
    - quaternion (numpy.ndarray): The quaternion representing the orientation.

    Returns:
    - forward_vector (numpy.ndarray): The calculated forward vector.
    """
    # Extract the components of the quaternion
    qw, qx, qy, qz = quaternion
    
    # Calculate the forward vector
    forward_vector = np.array([
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx**2 + qy**2)
        ])
    
    # Normalize the forward vector
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    
    return forward_vector

def calculate_forward_vector2(quaternion):
    """
    Calculate the forward vector based on a quaternion representing the orientation.

    Parameters:
    - quaternion (numpy.ndarray): The quaternion representing the orientation.

    Returns:
    - forward_vector (numpy.ndarray): The calculated forward vector.
    """
    w, x, y, z = quaternion
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    forward_vector = R[2]

    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    
    return forward_vector

def retruncate(date, terrain, trial, start, end):
    """
    Truncate the data based on start and end indices.

    Parameters:
    - date (str): The date of the data.
    - terrain (str): The terrain of the data.
    - trial (str): The trial of the data.
    - start (int): The start index.
    - end (int): The end index.
    """
    # Generate the filename based on the input parameters
    filename = fg.filename_clean_data(date, terrain, trial)
    
    # Read the clean data from the CSV file
    raw_test = pd.read_csv(filename)
    
    # Truncate the data based on the start and end indices
    if end != 9999:
        raw_test = raw_test.iloc[start:end]
    else:
        raw_test = raw_test.iloc[start:]
    # Save the truncated data back to the CSV file
    raw_test.to_csv(filename, index=False)