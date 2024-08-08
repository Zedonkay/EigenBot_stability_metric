import numpy as np
import pandas as pd
import filename_generation as fg
from matplotlib import cm

import matplotlib.pyplot as plt

# Function to import data from a CSV file
def import_data(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter=',')
    
    # Extract the time offset from the first row of the DataFrame
    time_offset = df.iloc[0, 0]
    
    # Extract the columns for timestamps, position, quaternion, velocity, acceleration, angular velocity, and angular acceleration
    raw_timestamps = df.iloc[:, 0]
    timestamps = raw_timestamps - time_offset
    pos_x = df.iloc[:, 1]
    pos_y = df.iloc[:, 2]
    pos_z = df.iloc[:, 3]
    quaternion = df.iloc[:, 4:8].values
    vel_x = df['vx']
    vel_y = df['vy']
    vel_z = df['vz']
    acc_x = df['ax']
    acc_y = df['ay']
    acc_z = df['az']
    wx = df['wx']
    wy = df['wy']
    wz = df['wz']
    aa_x = df['aa_x']
    aa_y = df['aa_y']
    aa_z = df['aa_z']
    roll = df['roll']
    pitch = df['pitch']
    yaw = df['yaw']
    
    return df, timestamps, pos_x, pos_y, pos_z, quaternion, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, wx, wy, wz, aa_x, aa_y, aa_z, roll, pitch, yaw

# Function to calculate displacement
def find_displacement(pos_x, pos_y):
    # Calculate the displacement at each timestamp
    displacements = []
    for i in range(len(pos_x)):
        displacements.append(np.linalg.norm(pos_x[i] - pos_x[len(pos_x) - 1]) + np.linalg.norm(pos_y[i] - pos_y[len(pos_y) - 1]))
    return displacements

# Function to convert quaternion to Euler angles
def quaternion_to_euler(quaternion):
    # Convert the quaternion representation to Euler angles (roll, pitch, yaw)
    roll = np.arctan2(2 * (quaternion[:, 0] * quaternion[:, 1] + quaternion[:, 2] * quaternion[:, 3]), 1 - 2 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2))
    pitch = np.arcsin(2 * (quaternion[:, 0] * quaternion[:, 2] - quaternion[:, 3] * quaternion[:, 1]))
    yaw = np.arctan2(2 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]), 1 - 2 * (quaternion[:, 2] ** 2 + quaternion[:, 3] ** 2))
    return roll, pitch, yaw

# Function to plot 2D linear state space
def plot_linear_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, terrain,object,test):
    # Create a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle(f"2D Linear State Space Plot for {object} on {terrain} terrain (test {test})")

    # Plot position X vs. timestamp
    axs[0, 0].plot(timestamps, pos_x)
    axs[0, 0].set_xlabel('Timestamp')
    axs[0, 0].set_ylabel('Position X')
    axs[0, 0].set_title('Position X vs. Timestamp')

    # Plot position Y vs. timestamp
    axs[0, 1].plot(timestamps, pos_y)
    axs[0, 1].set_xlabel('Timestamp')
    axs[0, 1].set_ylabel('Position Y')
    axs[0, 1].set_title('Position Y vs. Timestamp')

    # Plot position Z vs. timestamp
    axs[0, 2].plot(timestamps, pos_z)
    axs[0, 2].set_xlabel('Timestamp')
    axs[0, 2].set_ylabel('Position Z')
    axs[0, 2].set_title('Position Z vs. Timestamp')

    # Plot velocity X vs. timestamp
    axs[1, 0].plot(timestamps, vel_x)
    axs[1, 0].set_xlabel('Timestamp')
    axs[1, 0].set_ylabel('Velocity X')
    axs[1, 0].set_title('Velocity X vs. Timestamp')

    # Plot velocity Y vs. timestamp
    axs[1, 1].plot(timestamps, vel_y)
    axs[1, 1].set_xlabel('Timestamp')
    axs[1, 1].set_ylabel('Velocity Y')
    axs[1, 1].set_title('Velocity Y vs. Timestamp')

    # Plot velocity Z vs. timestamp
    axs[1, 2].plot(timestamps, vel_z)
    axs[1, 2].set_xlabel('Timestamp')
    axs[1, 2].set_ylabel('Velocity Z')
    axs[1, 2].set_title('Velocity Z vs. Timestamp')

    # Plot acceleration X vs. timestamp
    axs[2, 0].plot(timestamps, acc_x)
    axs[2, 0].set_xlabel('Timestamp')
    axs[2, 0].set_ylabel('Acceleration X')
    axs[2, 0].set_title('Acceleration X vs. Timestamp')

    # Plot acceleration Y vs. timestamp
    axs[2, 1].plot(timestamps, acc_y)
    axs[2, 1].set_xlabel('Timestamp')
    axs[2, 1].set_ylabel('Acceleration Y')
    axs[2, 1].set_title('Acceleration Y vs. Timestamp')

    # Plot acceleration Z vs. timestamp
    axs[2, 2].plot(timestamps, acc_z)
    axs[2, 2].set_xlabel('Timestamp')
    axs[2, 2].set_ylabel('Acceleration Z')
    axs[2, 2].set_title('Acceleration Z vs. Timestamp')

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(fg.store_clean_data(terrain,object,test) + "2d_linear_state_space.png")
    plt.clf()
    plt.close()

# Function to plot 2D angular state space
def plot_angular_2d(timestamps, roll, pitch, yaw, wx, wy, wz, aa_x, aa_y, aa_z, terrain,object,test):
    # Create a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle(f"2D Angular State Space Plot for {object} on {terrain} terrain (test {test})")

    # Plot roll vs. timestamp
    axs[0, 0].plot(timestamps, roll)
    axs[0, 0].set_xlabel('Timestamp')
    axs[0, 0].set_ylabel('Roll')
    axs[0, 0].set_title('Roll vs. Timestamp')

    # Plot pitch vs. timestamp
    axs[0, 1].plot(timestamps, pitch)
    axs[0, 1].set_xlabel('Timestamp')
    axs[0, 1].set_ylabel('Pitch')
    axs[0, 1].set_title('Pitch vs. Timestamp')

    # Plot yaw vs. timestamp
    axs[0, 2].plot(timestamps, yaw)
    axs[0, 2].set_xlabel('Timestamp')
    axs[0, 2].set_ylabel('Yaw')
    axs[0, 2].set_title('Yaw vs. Timestamp')

    # Plot angular velocity X vs. timestamp
    axs[1, 0].plot(timestamps, wx)
    axs[1, 0].set_xlabel('Timestamp')
    axs[1, 0].set_ylabel('Angular Velocity X')
    axs[1, 0].set_title('Angular Velocity X vs. Timestamp')

    # Plot angular velocity Y vs. timestamp
    axs[1, 1].plot(timestamps, wy)
    axs[1, 1].set_xlabel('Timestamp')
    axs[1, 1].set_ylabel('Angular Velocity Y')
    axs[1, 1].set_title('Angular Velocity Y vs. Timestamp')

    # Plot angular velocity Z vs. timestamp
    axs[1, 2].plot(timestamps, wz)
    axs[1, 2].set_xlabel('Timestamp')
    axs[1, 2].set_ylabel('Angular Velocity Z')
    axs[1, 2].set_title('Angular Velocity Z vs. Timestamp')

    # Plot angular acceleration X vs. timestamp
    axs[2, 0].plot(timestamps, aa_x)
    axs[2, 0].set_xlabel('Timestamp')
    axs[2, 0].set_ylabel('Angular Acceleration X')
    axs[2, 0].set_title('Angular Acceleration X vs. Timestamp')

    # Plot angular acceleration Y vs. timestamp
    axs[2, 1].plot(timestamps, aa_y)
    axs[2, 1].set_xlabel('Timestamp')
    axs[2, 1].set_ylabel('Angular Acceleration Y')
    axs[2, 1].set_title('Angular Acceleration Y vs. Timestamp')

    # Plot angular acceleration Z vs. timestamp
    axs[2, 2].plot(timestamps, aa_z)
    axs[2, 2].set_xlabel('Timestamp')
    axs[2, 2].set_ylabel('Angular Acceleration Z')
    axs[2, 2].set_title('Angular Acceleration Z vs. Timestamp')

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(fg.store_clean_data(terrain,object,test) + "2d_angular_state_space.png")
    plt.clf()
    plt.close()

# Function to plot 3D phase space plot for velocity
def plot_3d_phase_space_vel(vel_x, vel_y, vel_z, terrain,object,test):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory in 3D space
    ax.plot(vel_x, vel_y, vel_z)
    ax.set_xlabel('Velocity X')
    ax.set_ylabel('Velocity Y')
    ax.set_zlabel('Velocity Z')
    plt.title(f'3D Phase Space Plot for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "3d_phase_space_vel.png")
    plt.clf()
    plt.close()

# Function to plot 3D phase space plot for position
def plot_3d_phase_space_pos(pos_x, pos_y, pos_z, terrain,object,test):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory in 3D space
    ax.plot(pos_x, pos_y, pos_z)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    plt.title(f'3D Phase Space Plot for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "3d_state_space.png")
    plt.clf()
    plt.close()

# Function to plot 3D Euler state space
def plot_3d_euler_state_space(timestamps, roll, pitch, yaw, terrain,object,test):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory in 3D space
    ax.plot(roll, pitch, yaw)
    ax.set_xlabel('Roll (rad)')
    ax.set_ylabel('Pitch (rad)')
    ax.set_zlabel('Yaw (rad)')
    plt.title(f'3D Euler State Space Plot for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "3d_euler_state_space.png")
    plt.clf()
    plt.close()

# Function to plot 3D angular velocities
def plot_3d_angular_velocities(timestamps, ang_x, ang_y, ang_z, terrain,object,test):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory in 3D space
    ax.plot(ang_x, ang_y, ang_z)
    ax.set_xlabel('Angular Velocity X')
    ax.set_ylabel('Angular Velocity Y')
    ax.set_zlabel('Angular Velocity Z')
    plt.title(f'3D Angular Velocity Plot for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "3d_angular_velocities.png")
    plt.clf()
    plt.close()

# Function to plot 3D angular accelerations
def plot_3d_angular_accelerations(timestamps, ang_x, ang_y, ang_z, terrain,object,test):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory in 3D space
    ax.plot(ang_x, ang_y, ang_z)
    ax.set_xlabel('Angular Acceleration X')
    ax.set_ylabel('Angular Acceleration Y')
    ax.set_zlabel('Angular Acceleration Z')
    plt.title(f'3D Angular Acceleration Plot for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "3d_angular_accelerations.png")
    plt.clf()
    plt.close()

# Function to plot gait cycle
def plot_gait(pitch,roll, terrain,object,test):
    # Create a 2D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Plot
    ax.plot(pitch, roll)
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Roll')
    plt.title(f'Gait Cycle for {object} on {terrain} terrain (test {test})')
    
    # Save and close the plot
    plt.savefig(fg.store_clean_data(terrain,object,test) + "gait.png")
    plt.clf()
    plt.close()

# Main function
def main(terrain,object,test):
    # Generate the file path based on the disturbance and control type
    file_path = fg.filename_clean(terrain,object,test)
    
    # Import data from the CSV file
    df, timestamps, pos_x, pos_y, pos_z, quaternion, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, wx, wy, wz, aa_x, aa_y, aa_z, roll, pitch, yaw = import_data(file_path)
    
    # Plot 2D linear state space
    plot_linear_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, terrain,object,test)

    # Plot 2D angular state space
    plot_angular_2d(timestamps, roll, pitch, yaw, wx, wy, wz, aa_x, aa_y, aa_z, terrain,object,test)
    
    # Plot 3D phase space for position
    plot_3d_phase_space_pos(pos_x, pos_y, pos_z, terrain,object,test)
    
    # Plot 3D phase space for velocity
    plot_3d_phase_space_vel(vel_x, vel_y, vel_z, terrain,object,test)
    
    # Plot 3D Euler state space
    plot_3d_euler_state_space(timestamps, roll, pitch, yaw, terrain,object,test)
    
    # Plot 3D angular velocities
    plot_3d_angular_velocities(timestamps, wx, wy, wz, terrain,object,test)
    
    # Plot 3D angular accelerations
    plot_3d_angular_accelerations(timestamps, aa_x, aa_y, aa_z, terrain,object,test)

    # Plot gait cycle
    plot_gait(pitch,roll, terrain,object,test)
