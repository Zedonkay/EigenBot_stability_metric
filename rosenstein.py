import numpy as np
from scipy.signal import welch

def reconstruction(data,tau,m):
    
    d = len(data)
    d = d - (m-1)*tau
    if(len(data.shape)==1):
        reconstructed_data=np.empty((d,m))
    else:
        reconstructed_data = np.empty((d,m*len(data[0])))
    for i in range(d):
        for j in range(m):
            if(len(data.shape)==1):
                reconstructed_data[i][j] = data[i+j*tau]
            else:
                for k in range(len(data[0])):
                    
                    reconstructed_data[i][j*len(data[0])+k] = data[i+j*tau][k]
    return reconstructed_data

def welch_method(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency

def find_closest_vectors(reconstructed_data,min_step,t_f):
    #find closest vectors for rosenstein method
    neighbors = []
    avg_dist = []
    neighbors_index = []
    for i in range(len(reconstructed_data)):
        closest_dist = -1
        ind = -1
        for j in range(len(reconstructed_data)-t_f):
           if(i!=j and abs(j-i)>min_step):
               dist = np.linalg.norm(reconstructed_data[i]-reconstructed_data[j])
               if closest_dist == -1 or dist < closest_dist:
                   ind = j
                   closest_dist = dist
        
        if(closest_dist>0 and closest_dist<1e308 and not np.isnan(closest_dist)):
            neighbors.append(np.log(closest_dist))
            neighbors_index.append(ind)
        elif closest_dist == 0:
            neighbors.append(0)
            neighbors_index.append(-500)
        else:
            print(closest_dist)
            neighbors_index.append(-500)
    return neighbors_index

def log_distance(reconstructed_data,neighbors_index,i) -> float:
    #calculate expected log distance for rosenstein method
    d_ji = []
    for j in range(len(reconstructed_data)-i):
        if(neighbors_index[j]==-500):
            print("error")
        else:
            if j+i<len(reconstructed_data) and neighbors_index[j]+i<len(reconstructed_data):
                d_ji.append(np.linalg.norm(reconstructed_data[neighbors_index[j] + i]- reconstructed_data[j + i]))
            else:
                print(j,i)
    d_ji = np.array(d_ji)
    return np.mean(np.log(d_ji))

def lyapunov(data,tau,m, min_steps, t_0, t_f,delta_t,force_minsteps):
    # rosenstein method for lyapunov exponents

    #reconstruction through time delay
    reconstructed_data = reconstruction(data,tau,m)
    if not force_minsteps:
            min_steps = welch_method(data)
            if min_steps%1 != 0:
                min_steps = int(min_steps)+1
            else:
                min_steps = int(min_steps)
    #find closest vectors
    neighbors_index = find_closest_vectors(reconstructed_data,min_steps,t_f)
    #calculate mean distance
    mean_log_distance = []
    times = []
    for i in range(t_0,t_f):
        mean_log_distance.append(log_distance(reconstructed_data,neighbors_index,i))
        times.append(i*delta_t)
    mean_log_distance = np.array(mean_log_distance)
    times = np.array(times)
    #calculate lyapunov exponents
    return times, mean_log_distance