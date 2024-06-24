import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def welch_method(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency
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


def expected_log_distance(reconstructed_data,neighbors_index,i) -> float:
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
def rosenstein_lyapunov(data,tau,m, min_steps, t_0, t_f,delta_t,force_minsteps):
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
        mean_log_distance.append(expected_log_distance(reconstructed_data,neighbors_index,i))
        times.append(i*delta_t)
    mean_log_distance = np.array(mean_log_distance)
    times = np.array(times)
    #calculate lyapunov exponents
    return times, mean_log_distance

def kantz_mean_distance(vector_addresses,reconstructed_data,i):
    #calculate mean distance for kantz method
    mean_distance = []
    for j in range(len(vector_addresses)-i):
        dist =0
        for k in vector_addresses[j]:
            dist += np.log(np.linalg.norm(reconstructed_data[k+i]-reconstructed_data[j+i]))
        mean_distance.append(dist)
    return np.mean(mean_distance)
def kantz_distance(vector_addresses,reconstructed_data, t_0, t_f, delta_t):
    mean_distance = []
    times = []
    scaling = []
    for i in range(0,t_f):
        mean_distance.append(kantz_mean_distance(vector_addresses,reconstructed_data,i))
        
    for i in range(t_0,t_f):
        times.append(i*delta_t)
        scaling.append(mean_distance[i]/mean_distance[0])
    return times, scaling
def find_epsilon_vectors(reconstructed_data,epsilon):
    vector_addresses=[]
    for i in range(len(reconstructed_data)):
        print(i)
        vector_locations = []
        for j in range(len(reconstructed_data)):
            if i != j:
                dist = np.linalg.norm(reconstructed_data[i]-reconstructed_data[j])
                if dist<epsilon: 
                    vector_locations.append(j)
        vector_addresses.append(vector_locations)
    return vector_addresses
def kantz_lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon):
    # kantz method for lyapunov exponents

    #reconstruct data
    reconstructed_data = reconstruction(data,tau,m)

    #find vectors within epsilon distance of each vector
    vector_addresses = find_epsilon_vectors(reconstructed_data,epsilon)

    #calculate log distance
    times, mean_distances = kantz_distance(vector_addresses,reconstructed_data,t_0,t_f,delta_t)
    mean_distances = np.array(mean_distances)
    times = np.array(times)
        
   
    return times, mean_distances

def plot_growth_factors(times, lyap_exponents,fn,control_type,frequency):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times, fn(times),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.show()
    plt.clf()
def plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents):
    ay = plt.scatter(centralised_frequencies,centralised_exponents,label="Centralised")
    plt.scatter(distributed_frequencies,distributed_exponents,label="Distributed")
    plt.xlabel("Frequency")
    plt.ylabel("Lyapunov Exponent")
    plt.legend()
    plt.show()
    plt.clf()
def exponent(tau,m,min_steps, t_0,t_f,delta_t, force_minsteps,epsilon, filename,frequency, frequencies,exponents,control_type):
    #load data and format
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values

    #calculate lyapunov exponents with rosenstein method and plot growth
    times, data = rosenstein_lyapunov(data,tau,m, min_steps, t_0, t_f,delta_t,force_minsteps)
    coef=np.polyfit(times,data,1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,frequency)

    #track exponents and frequencies
    exponents.append(coef[0])
    frequencies.append(frequency)

    #store times and data in csv

def main():
    #parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    t_0 =80
    t_f =150
    epsilon= 0.01
    
    exponents = []
    frequencies = []
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,epsilon, "2_raw_data/official tests/centralised/odometry_data_centralised_test_10hz_1.csv",0,frequencies,exponents,"centralised")

   
    
if __name__ == "__main__":
    main()  