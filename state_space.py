import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import filename_generation as fg
import matplotlib.pyplot as plt

def import_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, delimiter=',')

    # Extract data columns
    time_offset = df.iloc[0, 0]
    raw_timestamps = df.iloc[:, 0]
    timestamps = raw_timestamps - time_offset
    pos_x = df[['px']].values
    pos_y = df[['py']].values
    pos_z = df[['pz']].values
    quaternion = df[['ox', 'oy', 'oz', 'ow']].values
    vel_x = df[['vel_x']].values
    vel_y = df[['vel_y']].values
    vel_z = df[['vel_z']].values


    return timestamps, pos_x, pos_y, pos_z, quaternion,vel_x, vel_y, vel_z

def find_displacement(pos_x,pos_y):
    displacements = []
    for i in range(len(pos_x)):
        displacements.append(np.linalg.norm(pos_x[i]-pos_x[len(pos_x)-1])+np.linalg.norm(pos_y[i]-pos_y[len(pos_y)-1]))
    return displacements 
def quaternion_to_euler(quaternion):
    # Convert quaternion to roll, pitch, yaw angles
    roll = np.arctan2(2*(quaternion[:, 0]*quaternion[:, 1] + quaternion[:, 2]*quaternion[:, 3]), 1 - 2*(quaternion[:, 1]**2 + quaternion[:, 2]**2))
    pitch = np.arcsin(2*(quaternion[:, 0]*quaternion[:, 2] - quaternion[:, 3]*quaternion[:, 1]))
    yaw = np.arctan2(2*(quaternion[:, 0]*quaternion[:, 3] + quaternion[:, 1]*quaternion[:, 2]), 1 - 2*(quaternion[:, 2]**2 + quaternion[:, 3]**2))
    return roll, pitch, yaw

def plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,frequency,test,control_type):
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
    plt.savefig(fg.store_clean_data(frequency,test,control_type) +"2d_state_space.png")
    plt.clf()
    plt.close()


#plot 3D phase space plot for velocity
def plot_3d_phase_space_vel(vel_x, vel_y, vel_z,frequency,test, control_type):
    # Plot 3D phase space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(vel_x, vel_y, vel_z)
    ax.set_xlabel('Velocity X')
    ax.set_ylabel('Velocity Y')
    ax.set_zlabel('Velocity Z')
    ax.set_title('3D Phase Space Plot')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+ "3d_phase_space_vel.png")
    plt.clf()
    plt.close

def plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency, test,control_type):
    # Plot 3D phase space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_x, pos_y, pos_z)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title('3D Phase Space Plot')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"3d_state_space.png")
    plt.clf()
    plt.close()

def plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,test,control_type):
    # Plot 3D Euler state space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(roll, pitch, yaw)
    ax.set_xlabel('Roll (rad)')
    ax.set_ylabel('Pitch (rad)')
    ax.set_zlabel('Yaw (rad)')
    ax.set_title('3D Euler State Space Plot')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"3d_euler_state_space.png")
    plt.clf()
    plt.close()

def plot_gait(displacements,pos_x,pos_y,pos_z,frequency,test,control_type):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(displacements, pos_z)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Z')
    ax.set_title('2D Gait Cycle Plot')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"gait.png")
    plt.clf()
    plt.close()



def main(frequency,control_type,test):
    file_path = fg.filename_clean(frequency,test,control_type)
    timestamps, pos_x, pos_y, pos_z, quaternion,vel_x, vel_y, vel_z = import_data(file_path)

    # Convert quaternion to roll, pitch, yaw angles
    roll, pitch, yaw = quaternion_to_euler(quaternion)

    # Plot 2D positions and angles
    plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, frequency,test,control_type)

    # Plot 3D phase space
    plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency,test,control_type)
    plot_3d_phase_space_vel(vel_x, vel_y, vel_z,frequency,test,control_type)

    # Plot 3D Euler state space
    plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,test,control_type)

    #plot 2d gait cycle
    displacements = find_displacement(pos_x,pos_y)
    plot_gait(displacements,pos_x,pos_y,pos_z,frequency,test,control_type)


if __name__ == "__main__":
    centralised = [[140, '1'], [180, '1'],
                   [180, '2'], [180, '3'], [220, '1'], [220, '2'],
                   [220, '3'], [260, '1'], [260, '2'], [260, '3'],
                   [320, '1'], [320, '2'],
                   [330, '1'], [350, '1']]

    distributed = [[100, '1'], [140, '1'], [140, '2'],
                   [180, '1'], [180, '2'], [220, '1'], [220, '2'],
                   [260, '1'], [260, '2'], [260, '3'], [300, '1'], [300, '2'], [300, '3'],
                   [350, '1'], [350, '2'], [370, '1'], [370, '2'], [400, '1'],
                   [400, '2'], [500, '1'], [600, '1'], [700, '1'], [800, '1'],
                   [1000, '1'], [1040, '1'], [1120, '1'], [1160, '1'], [1200, '1'],
                   [1440, '1'], [1600, '1'], [1800, '1'], [2000, '1']]
    print("running centralised")
    for i in centralised:
        main(i[0], 'centralised', i[1])
    print("running distributed")
    for i in distributed:
        main(i[0], 'distributed', i[1])
    print("done")
