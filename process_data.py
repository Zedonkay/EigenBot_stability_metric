import pandas as pd
import numpy as np
import filename_generation as fg
import pickle
import scipy

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
    filename = fg.filename_clean_data(date, terrain, trial)
    
    # Read the raw data from the CSV file
    df = pd.read_csv(filename)
    
    df['timestamp'] = df['stamp'] - df['stamp'][0]
    df['timestamp'] = df['timestamp'] / 1e9

    positions = df[['px', 'py', 'pz']].values

    if 'terrain' in terrain and date!="Simulation":
        df['lyap_z']=terrain_fit(date,terrain,trial, positions,df['timestamp'].values)
    elif date=="Simulation" and 'terrain' in terrain:
        df['lyap_z'] = terrain_fit_sim(date,terrain,trial, positions,df['timestamp'].values)
    else:
        df['lyap_z'] = positions[:, 2]
        
    #Write the new positions to the dataframe
    df['px'] = positions[:, 0]
    df['py'] = positions[:, 1]
    df['pz'] = positions[:, 2]
        

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
    
    # Calculate forward velocity
    positions = df[['px', 'py', 'pz']].values
    df['forward_velocity'] = calculate_forward_velocity(positions, quaternions, times,1)
    df['forward_velocity_1']=calculate_forward_velocity(positions, quaternions, times,2)

    data_np = df[['roll', 'pitch']].to_numpy()

    # Define mean values
    mean_values = np.zeros((1, 2))

    # Compute standard deviation using NumPy (ensuring mean is passed correctly)
    deviation_values = np.std(data_np, axis=0, mean=mean_values)

    # Compute norm of deviation values
    deviation = np.linalg.norm(deviation_values)
    df['deviation'] = deviation
    print("deviation:", deviation)
    # Save the processed data to a new CSV file

    df.to_csv(fg.filename_clean_data(date, terrain, trial), index=False)

def calculate_forward_velocity(positions, quaternions, times,attempt):
    """
    Calculate the forward velocity of the robot based on positions and quaternions.

    Parameters:
    - positions (numpy.ndarray): The positions of the robot.
    - quaternions (numpy.ndarray): The quaternions representing the orientation of the robot.
    - times (numpy.ndarray): The corresponding times.

    Returns:
    - forward_velocities (numpy.ndarray): The calculated forward velocities.
    """
    forward_velocities = []
    for i in range(len(positions)):
        if i == 0:
            forward_velocities.append(0)
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
        forward_vector = calculate_forward_vector(current_quaternion,attempt)
        
        # Calculate the displacement vector between the current and previous positions
        displacement_vector = current_position - previous_position
        
        # # Calculate the forward displacement by projecting the displacement vector onto the forward vector
        forward_displacement = np.dot(displacement_vector, forward_vector)
        # Calculate the forward velocity
        forward_velocity = forward_displacement / (current_time - previous_time)
            

        forward_velocities.append(forward_velocity)
    
    return np.array(forward_velocities)
def calculate_deviation(positions, quaternions, times):
    """
    Calculate the deviation of the robots orientation from its median orientation.
    Parameters:
    - positions (numpy.ndarray): The positions of the robot.
    - quaternions (numpy.ndarray): The quaternions representing the orientation of the robot.
    - times (numpy.ndarray): The corresponding times.
    Returns:
    - deviation (numpy.ndarray): The calculated deviation of the robots orientation.
    """
    # Calculate the median quaternion
    median_quaternion = np.median(quaternions, axis=0)

    # Calculate the deviation of each quaternion from the median quaternion
    deviation = np.linalg.norm(quaternions - median_quaternion, axis=1)
    # Normalize the deviation by the median quaternion
    deviation = deviation / np.linalg.norm(median_quaternion)
    return deviation
def calculate_forward_vector(quaternion,attempt):
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
   
    if attempt==1:
        forward_vector = R[:, 0]
    else:
        forward_vector = R[0]
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    
    return forward_vector

def truncate(date, terrain, trial, start, end):
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
    raw_test = pd.read_csv(fg.filename_raw_data(date, terrain, trial))
    
    # Truncate the data based on the start and end indices
    if end != 9999:
        raw_test = raw_test.iloc[start:end]
    else:
        raw_test = raw_test.iloc[start:]
    
    # Save the truncated data back to the CSV file
    raw_test.to_csv(filename, index=False)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model
    
def predict_z(model, x, y):
    z = model(x, y)
    return z

def terrain_fit(date,terrain,trial, positions,timestamps):
    """
    Fit the data to the terrain.

    Parameters:
    - positions (numpy.ndarray): The input position data.

    Returns:
    - positions (numpy.ndarray): The position data fitted to the terrain.
    """
    terrain_start_x = positions[0][0] + .67
    terrain_end_x = positions[0][0] + .67 + 2
    terrain_end_y = positions[0][1] + .92
    terrain_start_y = positions[0][1] - .6
    terrain_timestamps = [] 
    model = load_model('4_models/rbf_model_bump.pkl')
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]
        if x >= terrain_start_x and x <= terrain_end_x and y >= terrain_start_y and y <= terrain_end_y:
            model_x = (x - terrain_start_x-1)*1000
            model_y = (y - terrain_start_y)*1000
            z = predict_z(model, model_x, model_y)
            z=z/1000
            positions[i][2] =positions[i][2] - z
            terrain_timestamps.append(timestamps[i])
    pd.DataFrame(terrain_timestamps).to_csv(fg.filename_store_data(date,terrain,trial)+'terrain_timestamps.csv',index=False)
    return positions[:,2]


def terrain_fit_sim(date,terrain,trial,positions,timestamps):
    data = np.loadtxt('4_models/sim_terrain.csv', delimiter=',')
    points = data[:, :2]
    values = data[:, 2]
    terrain_timestamps = []
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]
        z=scipy.interpolate.griddata(points, values, (x, y), method='linear')
        positions[i][2] =positions[i][2] - z
        terrain_timestamps.append(timestamps[i])
    pd.DataFrame(terrain_timestamps).to_csv(fg.filename_store_data(date,terrain,trial)+'terrain_timestamps.csv',index=False)
    return positions[:,2]