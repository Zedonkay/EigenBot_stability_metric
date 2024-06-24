import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import math

def import_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, delimiter=',')

    # Extract data columns
    time_offset = df.iloc[0, 0]
    raw_timestamps = df.iloc[:, 0]
    timestamps = raw_timestamps - time_offset
    pos_x = df.iloc[:, 1]
    pos_y = df.iloc[:, 2]
    pos_z = df.iloc[:, 3]
    quaternion = df.iloc[:, 4:8].values

    # # Truncate arrays up to timestamps where timestamps = 4.53
    # begin_trunc_index = np.argmax(timestamps >= 5.31)
    # end_trunc_index = np.argmax(timestamps >= 20.0)
    # timestamps = timestamps[begin_trunc_index:end_trunc_index]
    # pos_x = pos_x[begin_trunc_index:end_trunc_index]
    # pos_y = pos_y[begin_trunc_index:end_trunc_index]
    # pos_z = pos_z[begin_trunc_index:end_trunc_index]
    # quaternion = quaternion[begin_trunc_index:end_trunc_index]

    # Print sizes of arrays after truncation
    print("Size of timestamps array after truncation:", timestamps.shape)
    print("Size of pos_x array after truncation:", pos_x.shape)
    print("Size of pos_y array after truncation:", pos_y.shape)
    print("Size of pos_z array after truncation:", pos_z.shape)
    print("Size of quaternion array after truncation:", quaternion.shape)

    return timestamps, pos_x, pos_y, pos_z, quaternion

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


def quaternion_to_euler(quaternion):
    roll = np.arctan2(2*(quaternion[:, 0]*quaternion[:, 1] + quaternion[:, 2]*quaternion[:, 3]), 1 - 2*(quaternion[:, 1]**2 + quaternion[:, 2]**2))
    pitch = np.arcsin(2*(quaternion[:, 0]*quaternion[:, 2] - quaternion[:, 3]*quaternion[:, 1]))
    yaw = np.arctan2(2*(quaternion[:, 0]*quaternion[:, 3] + quaternion[:, 1]*quaternion[:, 2]), 1 - 2*(quaternion[:, 2]**2 + quaternion[:, 3]**2))
    return roll, pitch, yaw

def plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,frequency,control_type):
    # Plot individual 2D plots for pos_x, pos_y, pos_z, roll, pitch, yaw
    fig, axs = plt.subplots(3, 3, figsize=(18, 10))
    axs[0, 0].plot(timestamps, pos_x)
    axs[0, 0].set_xlabel('Timestamp')
    axs[0, 0].set_ylabel('Position X')
    axs[0, 0].set_title('Position X vs. Timestamp')

    axs[0, 1].plot(timestamps, pos_y)
    axs[0, 1].set_xlabel('Timestamp')
    axs[0, 1].set_ylabel('Position Y')
    axs[0, 1].set_title('Position Y vs. Timestamp')

    axs[0, 2].plot(timestamps, pos_z)
    axs[0, 2].set_xlabel('Timestamp')
    axs[0, 2].set_ylabel('Position Z')
    axs[0, 2].set_title('Position Z vs. Timestamp')

    axs[1, 0].plot(timestamps, vel_x)
    axs[1, 0].set_xlabel('Timestamp')
    axs[1, 0].set_ylabel('Velocity X')
    axs[1, 0].set_title('Velocity X vs. Timestamp')

    axs[1, 1].plot(timestamps, vel_y)
    axs[1, 1].set_xlabel('Timestamp')
    axs[1, 1].set_ylabel('Velocity Y')
    axs[1, 1].set_title('Velocity Y vs. Timestamp')

    axs[1, 2].plot(timestamps, vel_z)
    axs[1, 2].set_xlabel('Timestamp')
    axs[1, 2].set_ylabel('Velocity Z')
    axs[1, 2].set_title('Velocity Z vs. Timestamp')

    # plot 2D angles on the third row of the same plot
    axs[2, 0].plot(timestamps, roll)
    axs[2, 0].set_xlabel('Timestamp')
    axs[2, 0].set_ylabel('Roll (rad)')
    axs[2, 0].set_title('Roll vs. Timestamp')

    axs[2, 1].plot(timestamps, pitch)
    axs[2, 1].set_xlabel('Timestamp')
    axs[2, 1].set_ylabel('Pitch (rad)')
    axs[2, 1].set_title('Pitch vs. Timestamp')

    axs[2, 2].plot(timestamps, yaw)
    axs[2, 2].set_xlabel('Timestamp')
    axs[2, 2].set_ylabel('Yaw (rad)')
    axs[2, 2].set_title('Yaw vs. Timestamp')

    plt.tight_layout()
    plt.savefig('6_Eigenbot/'+control_type+"/"+str(frequency)+"Hz/"+str(frequency)+"Hz_2d_state_space.png")
    plt.clf()
    plt.close()


#plot 3D phase space plot for velocity
def plot_3d_phase_space_vel(vel_x, vel_y, vel_z,frequency,control_type):
    # Plot 3D phase space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(vel_x, vel_y, vel_z)
    ax.set_xlabel('Velocity X')
    ax.set_ylabel('Velocity Y')
    ax.set_zlabel('Velocity Z')
    ax.set_title('3D Phase Space Plot')
    plt.savefig('6_Eigenbot/'+control_type+"/"+str(frequency)+"Hz/"+str(frequency)+"Hz_3d_phase_space_vel.png")
    plt.clf()

def plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency,control_type):
    # Plot 3D phase space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_x, pos_y, pos_z)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title('3D Phase Space Plot')
    plt.savefig('6_Eigenbot/'+control_type+"/"+str(frequency)+"Hz/"+str(frequency)+"Hz_3d_state_space.png")
    plt.clf()

def plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,control_type):
    # Plot 3D Euler state space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(roll, pitch, yaw)
    ax.set_xlabel('Roll (rad)')
    ax.set_ylabel('Pitch (rad)')
    ax.set_zlabel('Yaw (rad)')
    ax.set_title('3D Euler State Space Plot')
    plt.savefig('6_Eigenbot/'+control_type+"/"+str(frequency)+"Hz/"+str(frequency)+"Hz_3d_euler_state_space.png")
    plt.clf()

def main(file_path,frequency,control_type):

    # Import data
    timestamps, pos_x, pos_y, pos_z, quaternion = import_data(file_path)

    vel_x, vel_y, vel_z = compute_velocities(pos_x, pos_y, pos_z, timestamps)
    # Convert quaternion to roll, pitch, yaw angles
    roll, pitch, yaw = quaternion_to_euler(quaternion)

    # Plot 2D positions and angles
    plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, frequency,control_type)

    # Plot 3D phase space
    plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency,control_type)
    plot_3d_phase_space_vel(vel_x, vel_y, vel_z,frequency,control_type)

    # Plot 3D Euler state space
    plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,control_type)

if __name__ == "__main__":
    print("Running for centralised data")
    main("1_clean_data/centralised/140Hz.csv",140,"centralised")
    print("140Hz done")
    main("1_clean_data/centralised/220Hz.csv",220,"centralised")
    print("220Hz done")
    main("1_clean_data/centralised/350Hz.csv",350,"centralised")
    print("350Hz done")
    main("1_clean_data/centralised/370Hz.csv",370,"centralised")
    print("370Hz done")
    main("1_clean_data/centralised/400Hz.csv",400,"centralised")
    print("400Hz done")

    print("Running for distributed data")
    main("1_clean_data/distributed/140Hz.csv",140,"distributed")
    print("140Hz done")
    main("1_clean_data/distributed/220Hz.csv",220,"distributed")
    print("220Hz done")
    main("1_clean_data/distributed/350Hz.csv",350,"distributed")
    print("350Hz done")
    main("1_clean_data/distributed/370Hz.csv",370,"distributed")
    print("370Hz done")
    main("1_clean_data/distributed/400Hz.csv",400,"distributed")
    print("400Hz done")
    main("1_clean_data/distributed/500Hz.csv",500,"distributed")
    print("500Hz done")
    main("1_clean_data/distributed/600Hz.csv",600,"distributed")
    print("600Hz done")
    main("1_clean_data/distributed/800Hz.csv",800,"distributed")
    print("800Hz done")
    main("1_clean_data/distributed/1000Hz.csv",1000,"distributed")
    print("1000Hz done")
    main("1_clean_data/distributed/1120Hz.csv",1120,"distributed")
    print("1120Hz done")
    main("1_clean_data/distributed/1200Hz.csv",1200,"distributed")
    print("1200Hz done")
    main("1_clean_data/distributed/1440Hz.csv",1440,"distributed")
    print("1440Hz done")
    main("1_clean_data/distributed/1800Hz.csv",1800,"distributed")
    print("1800Hz done")
    main("1_clean_data/distributed/2000Hz.csv",2000,"distributed")
    print("2000Hz done")
