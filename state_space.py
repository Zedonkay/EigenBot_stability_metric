import numpy as np
import pandas as pd
import filename_generation as fg
import matplotlib.pyplot as plt
from matplotlib import cm


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
    vel_x=df['vx']
    vel_y=df['vy']
    vel_z=df['vz']
    acc_x=df['ax']
    acc_y=df['ay']
    acc_z=df['az']

    return df, timestamps, pos_x, pos_y, pos_z, quaternion, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z
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
def find_lines(pos_x,tolerance):
    lines = []
    i=0
    while i<len(pos_x)-2:
        if pos_x[i+1]-pos_x[i]>tolerance:
            current_line = []
            current_line.append(i)
            for j in range(i+1,len(pos_x)-1):
                if pos_x[j+1]-pos_x[j]<0:
                    current_line.append(j)
                    break
            lines.append(current_line)
            i = j
        i+=1
    if(len( lines[len(lines)-1])<2):
        lines.pop()
    p_1 = (lines[0][0]+lines[0][1])/2
    p_2 = (lines[len(lines)-1][0]+lines[len(lines)-1][1])/2
    return int(round(p_1)),int(round(p_2))
def find_top_peaks(pos_z):
    peaks = []
    for i in range(1,len(pos_z)-1):
        if pos_z[i]>pos_z[i-1] and pos_z[i]>pos_z[i+1]:
            if(len(pos_z)-i>50):
                next_positions = pos_z[i:i+50]
            else:
                next_positions = pos_z[i:]
            if(i>50):
                previous_positions = pos_z[i-50:i]
            else:
                previous_positions = pos_z[:i]
            if len(next_positions[next_positions>pos_z[i]])==0 and len(previous_positions[previous_positions>pos_z[i]])==0:
                peaks.append(i)
    peaks.append(len(pos_z)-1)
    return peaks  
def find_bottom_peaks(pos_z):
    peaks = []
    for i in range(1,len(pos_z)-1):
        if pos_z[i]<pos_z[i-1] and pos_z[i]<pos_z[i+1]:
            if(len(pos_z)-i>50):
                next_positions = pos_z[i:i+50]
            else:
                next_positions = pos_z[i:]
            if(i>50):
                previous_positions = pos_z[i-50:i]
            else:
                previous_positions = pos_z[:i]
            if len(next_positions[next_positions<pos_z[i]])==0 and len(previous_positions[previous_positions<pos_z[i]])==0:
                peaks.append(i)
    
    return peaks
def compute_midpoints(bottom_peaks,top_peaks,pos_z):
    midpoints = []
    for i in range(min(len(top_peaks)-1,len(bottom_peaks)-1)):
        midpoints.append(int((top_peaks[i]+bottom_peaks[i])/2))
    midpoints.append(len(pos_z)-1)
    midpoints = np.array(midpoints)
    np.insert(midpoints,0,0)
    return midpoints


def rotate_data(data,min,max,initial_x,initial_z):
    vector = [data[max][0]-data[min][0],
              data[max][1]-data[min][1],
              data[max][2]-data[min][2]]
    angle = -np.arctan(vector[0]/vector[1])
    rotated_data = []
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    for i in range(min,max):
        current_data= np.dot(rotation_matrix,np.array([data[i][0],data[i][1],data[i][2]]))
        current_data[0] = current_data[0]-initial_x+data[min][0]
        current_data[2] = current_data[2]-initial_z+data[min][2]
        rotated_data.append(current_data)
    rotated_data = np.array(rotated_data)
    return rotated_data    
def compute_plotting_points(data,bottom_peaks):
    initial_x = data[bottom_peaks[0]][0]
    initial_z= data[bottom_peaks[0]][2]
    for i in range(len(bottom_peaks)-1):
        if i == 0:
            rotated_data = rotate_data(data,bottom_peaks[i],bottom_peaks[i+1],initial_x,initial_z)
        else:
            rotated_data = np.concatenate((rotated_data,rotate_data(data,bottom_peaks[i],bottom_peaks[i+1],initial_x,initial_z)))
    rotated_data = np.array(rotated_data)
    return rotated_data[:,0],rotated_data[:,1],rotated_data[:,2]

def plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,acc_x,acc_y,acc_z, 
            roll, pitch, yaw,frequency,test,control_type,midpoints):
    # Plot individual 2D plots for pos_x, pos_y, pos_z, roll, pitch, yaw
    fig, axs = plt.subplots(4, 3, figsize=(18, 10))
    if control_type == "centralised":
        fig.suptitle(f'2D State Space Plots for Centralised Control at {frequency} Hz')
    else:
        fig.suptitle(f'2D State Space Plots for Distributed Control at {frequency} Hz')


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

    # plot 2D accelerations on the third row of the same plot
    axs[2, 0].plot(timestamps, acc_x)
    axs[2, 0].set_xlabel('Timestamp')
    axs[2, 0].set_ylabel('Acceleration X')
    axs[2, 0].set_title('Acceleration X vs. Timestamp')

    axs[2, 1].plot(timestamps, acc_y)
    axs[2, 1].set_xlabel('Timestamp')
    axs[2, 1].set_ylabel('Acceleration Y')
    axs[2, 1].set_title('Acceleration Y vs. Timestamp')

    axs[2, 2].plot(timestamps, acc_z)
    axs[2, 2].set_xlabel('Timestamp')
    axs[2, 2].set_ylabel('Acceleration Z')
    axs[2, 2].set_title('Acceleration Z vs. Timestamp')

    # plot 2D angles on the fourth row of the same plot
    axs[3, 0].plot(timestamps, roll)
    axs[3, 0].set_xlabel('Timestamp')
    axs[3, 0].set_ylabel('Roll (rad)')
    axs[3, 0].set_title('Roll vs. Timestamp')

    axs[3, 1].plot(timestamps, pitch)
    axs[3, 1].set_xlabel('Timestamp')
    axs[3, 1].set_ylabel('Pitch (rad)')
    axs[3, 1].set_title('Pitch vs. Timestamp')

    axs[3, 2].plot(timestamps, yaw)
    axs[3, 2].set_xlabel('Timestamp')
    axs[3, 2].set_ylabel('Yaw (rad)')
    axs[3, 2].set_title('Yaw vs. Timestamp')



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
    if control_type == "centralised":
        plt.title(f'3D Phase Space Plot for Centralised Control at {frequency} Hz')
    else:
        plt.title(f'3D Phase Space Plot for Distributed Control at {frequency} Hz')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+ "3d_phase_space_vel.png")
    plt.clf()
    plt.close

def plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency, test,control_type,bottom_peaks,top_peaks):
    # Plot 3D phase space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_x, pos_y, pos_z)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    if control_type == "centralised":
        plt.title(f'3D Phase Space Plot for Centralised Control at {frequency} Hz')
    else:
        plt.title(f'3D Phase Space Plot for Distributed Control at {frequency} Hz')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"3d_state_space.png")
    plt.clf()
    plt.close()
def plot_rotated_helix(pos_x, pos_y, pos_z,frequency,test,control_type,bottom_peaks):
    # Plot 3D phase space plot
    bottom_peaks = np.array(bottom_peaks)
    bottom_peaks = bottom_peaks[bottom_peaks<min(len(pos_x),len(pos_y),len(pos_z))]

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Rotated Helix Plot')

    axs[0, 0].plot(pos_x, pos_y)
    axs[0, 0].scatter(pos_x[bottom_peaks],pos_y[bottom_peaks])
    axs[0, 0].set_xlabel('Position X')
    axs[0, 0].set_ylabel('Position Y')
    axs[0, 0].set_title('Position X vs. Position Y')

    axs[0, 1].plot(pos_x, pos_z)
    axs[0,1].scatter(pos_x[bottom_peaks],pos_z[bottom_peaks])
    axs[0, 1].set_xlabel('Position X')
    axs[0, 1].set_ylabel('Position Z')
    axs[0, 1].set_title('Position X vs. Position Z')

    axs[1, 0].plot(pos_y, pos_z)
    axs[1,0].scatter(pos_y[bottom_peaks],pos_z[bottom_peaks])
    axs[1, 0].set_xlabel('Position Y')
    axs[1, 0].set_ylabel('Position Z')
    axs[1, 0].set_title('Position Y vs. Position Z')

    axs[1,1]=plt.subplot(224,projection='3d')
    axs[1,1].scatter(pos_x[bottom_peaks],pos_y[bottom_peaks],pos_z[bottom_peaks])
    axs[1,1].plot(pos_x,pos_y,pos_z)
    axs[1,1].set_xlabel('Position X')
    axs[1,1].set_ylabel('Position Y')
    axs[1,1].set_zlabel('Position Z')

    plt.tight_layout()
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"rotated_helix.png")

def plot_original_helix(pos_x, pos_y, pos_z,frequency,test,control_type,bottom_peaks):
    # Plot 3D phase space plot
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Original Helix Plot')

    axs[0, 0].plot(pos_x, pos_y)
    axs[0, 0].scatter(pos_x[bottom_peaks],pos_y[bottom_peaks])
    axs[0, 0].set_xlabel('Position X')
    axs[0, 0].set_ylabel('Position Y')
    axs[0, 0].set_title('Position X vs. Position Y')

    axs[0, 1].plot(pos_x, pos_z)
    axs[0,1].scatter(pos_x[bottom_peaks],pos_z[bottom_peaks])
    axs[0, 1].set_xlabel('Position X')
    axs[0, 1].set_ylabel('Position Z')
    axs[0, 1].set_title('Position X vs. Position Z')

    axs[1, 0].plot(pos_y, pos_z)
    axs[1,0].scatter(pos_y[bottom_peaks],pos_z[bottom_peaks])
    axs[1, 0].set_xlabel('Position Y')
    axs[1, 0].set_ylabel('Position Z')
    axs[1, 0].set_title('Position Y vs. Position Z')

    axs[1,1]=plt.subplot(224,projection='3d')
    axs[1,1].scatter(pos_x[bottom_peaks],pos_y[bottom_peaks],pos_z[bottom_peaks])
    axs[1,1].plot(pos_x,pos_y,pos_z)
    axs[1,1].set_xlabel('Position X')
    axs[1,1].set_ylabel('Position Y')
    axs[1,1].set_zlabel('Position Z')

    plt.tight_layout()
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"original_helix.png")



def plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,test,control_type):
    # Plot 3D Euler state space plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(roll, pitch, yaw)
    ax.set_xlabel('Roll (rad)')
    ax.set_ylabel('Pitch (rad)')
    ax.set_zlabel('Yaw (rad)')
    if control_type == "centralised":
        plt.title(f'3D Euler State Space Plot for Centralised Control at {frequency} Hz')
    else:
        plt.title(f'3D Euler State Space Plot for Distributed Control at {frequency} Hz')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"3d_euler_state_space.png")
    plt.clf()
    plt.close()

def plot_gait(plot_x,plot_y,plot_z,bottom_peaks,frequency,test,control_type):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(bottom_peaks)-1):
        color = cm.rainbow(np.linspace(0,1,bottom_peaks[i+1]-bottom_peaks[i]))
        for j in range(bottom_peaks[i],bottom_peaks[i+1]-1,4):
            ax.plot(plot_x[j:j+5],plot_y[j:j+5],color=color[j-bottom_peaks[i]])
    ax.set_xlabel('Transverse Plane')
    ax.set_ylabel('Frontal Plane')
    if control_type == "centralised":
        plt.title(f'Gait Cycle for Centralised Control at {frequency} Hz')
    else:
        plt.title(f'Gait Cycle for Distributed Control at {frequency} Hz')
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"gait.png")
    plt.clf()
    plt.close()

def main(frequency,control_type,test):
    # Generate file path
    file_path = fg.filename_clean(frequency,test,control_type)

    # Import data
    df, timestamps, pos_x, pos_y, pos_z, quaternion, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = import_data(file_path)

    # Convert quaternion to roll, pitch, yaw angles
    roll, pitch, yaw = quaternion_to_euler(quaternion)
    
   

    # find peaks
    top_peaks = find_top_peaks(pos_z)
    bottom_peaks = find_bottom_peaks(pos_z)
    midpoints = compute_midpoints(bottom_peaks,top_peaks,pos_z)

    #compute plotting points
    plot_x,plot_y,plot_z = compute_plotting_points(np.column_stack((pos_x,pos_y,pos_z)),bottom_peaks)

    

    # Plot 2D positions and angles
    plot_2d(timestamps, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
            acc_x,acc_y,acc_z,roll, pitch, yaw, frequency,test,control_type,midpoints)

    # Plot 3D phase space
    plot_3d_phase_space_pos(pos_x, pos_y, pos_z,frequency,test,control_type,bottom_peaks,top_peaks)
    plot_3d_phase_space_vel(vel_x, vel_y, vel_z,frequency,test,control_type)

    # Plot 3D Euler state space
    plot_3d_euler_state_space(timestamps, roll, pitch, yaw,frequency,test,control_type)
    
    #plot 2d gait cycle
    plot_gait(plot_x,plot_y,plot_z,bottom_peaks, frequency,test,control_type)

    #plot rotated data
    plot_rotated_helix(plot_x,plot_y,plot_z,frequency,test,control_type,bottom_peaks)

    #plot original data
    plot_original_helix(pos_x, pos_y, pos_z,frequency,test,control_type,bottom_peaks)