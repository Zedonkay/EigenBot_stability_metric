import matplotlib.pyplot as plt
import math
import numpy as np

def rotate_data(data,min,max,initial_x):
    vector = [data[max][0]-data[min][0],
              data[max][1]-data[min][1],
              data[max][2]-data[min][2]]
    angle = np.arctan2(vector[1],vector[0])
    rotated_data = []
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    for i in range(min,max):
        current_data= np.dot(rotation_matrix,np.array([data[i][0],data[i][1],data[i][2]]))
        current_data[0] = current_data[0]-initial_x
        rotated_data.append(current_data)
    rotated_data = np.array(rotated_data)
    return rotated_data    

def compute_plotting_points(data,bottom_peaks):
    initial_x = data[0][0]
    for i in range(len(bottom_peaks)-1):
        if i == 0:
            rotated_data = rotate_data(data,bottom_peaks[i],bottom_peaks[i+1],initial_x)
        else:
            rotated_data = np.concatenate((rotated_data,rotate_data(data,bottom_peaks[i],bottom_peaks[i+1],initial_x)))
    rotated_data = np.array(rotated_data)
    return rotated_data[:,0],rotated_data[:,1],rotated_data[:,2]
def bottom_peaks(data):
    return np.where(data[:,2] == np.min(data[:,2]))[0]
    
def helix_plot(i):
    x=np.cos(math.radians(i))
    y=np.sin(math.radians(i))
    z=i/360
    W = [1,1,0]
    W = W/np.linalg.norm(W)
    U = np.cross(W,[0,0,1])
    U = U/np.linalg.norm(U)
    V = np.cross(W,U)
    V = V/np.linalg.norm(V)
    Matrix = np.array([U,V,W])
    return np.dot([x,y,z],Matrix)
def generate_values():
    helix_values = []
    values = np.arange(0,1080,5)
    for i in values:
        position = helix_plot(i)
        helix_values.append(position)
    helix_values = np.array(helix_values)
    return helix_values

def plot_helix(helix_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(helix_values)):
        ax.scatter(helix_values[i][0], helix_values[i][1], helix_values[i][2])
    fig.suptitle('Helix Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_points(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    fig.suptitle('Helix Plot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    helix_values = generate_values()
    plot_helix(helix_values)
    low_peaks = bottom_peaks(helix_values)
    x,y,z = compute_plotting_points(helix_values,low_peaks)
    plot_points(x,y,z)


if __name__ == "__main__":
    main()