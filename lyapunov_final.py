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
    print(mean_frequency)
    return 1 / mean_frequency

def reconstruction(data,tau,m):

    d = len(data)
    d = d - (m-1)*tau
    if(len(data.shape)==1):
        reconstructed_data=np.zeros((d,m))
    else:
        reconstructed_data = np.zeros((d,m*len(data[0])))
    for i in range(d):
        for j in range(m):
            if(len(data.shape)==1):
                reconstructed_data[i][j] = data[i+j*tau]
            else:
                for k in range(len(data[0])):
                    reconstructed_data[i][j*len(data[0])+k] = data[i+j*tau][k]
    return reconstructed_data



def find_closest_vectors(reconstructed_data,min_step):
    #find closest vectors for rosenstein method
    neighbors = []
    avg_dist = []
    neighbors_index = []
    for i in range(len(reconstructed_data)):
        print("checking for vector closest to vector ",i)
        closest_dist = -1
        ind = -1
        for j in range(len(reconstructed_data)):
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
    neighbors_index = np.array(neighbors_index)
    return neighbors_index


def dist(i, reconstructed_data,neighbors_index):
    d_neighbors_indexi = np.array([np.linalg.norm(reconstructed_data[neighbors_index[k] + i], reconstructed_data[k + i]) for k in range(len(reconstructed_data) - i)])
    return np.mean(np.log(d_neighbors_indexi))
def rosenstein_lyapunov(data,tau,m, min_steps, t_0, t_f,delta_t,force_minsteps):
    # rosenstein method for lyapunov exponents

    #reconstruction through time delay
    reconstructed_data = reconstruction(data,tau,m)
    print("data has been reconstructed")
    if not force_minsteps:
            min_steps = welch_method(data)
            if min_steps%1 != 0:
                min_steps = int(min_steps)+1
            else:
                min_steps = int(min_steps)


    print(min_steps)
    #find closest vectors
    neighbors_index = find_closest_vectors(reconstructed_data,min_steps)
    print(j)
    mean_log_distance = np.array([dist(i, reconstructed_data, neighbors_index) for i in range(t_0, t_f)])
    times = np.arange(t_0, t_f)*delta_t
    print(np.polyfit(times,mean_log_distance,1))
    return times, mean_log_distance
    
def plot_growth_factors(times, lyap_exponents):
    """Plot Lyapunov exponents."""
    ax = plt.plot(lyap_exponents,label="Average divergence", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.show()

def exponent(tau,m,min_steps, t_0,t_f,delta_t,filename, force_minsteps):
    #load data and format
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values
    
    

    
    #calculate lyapunov exponents with rosenstein method and plot growth
    times, data = rosenstein_lyapunov(data,tau,m,min_steps,t_0,t_f,delta_t,force_minsteps)
    plot_growth_factors(times, data)
   

def main():
    #parameters
    filename = "1_clean_data/centralised/140Hz.csv"
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    t_0 =0

    # must be greater than t_0 and less than the length of the data reconstructed length
    # length of reconsturctd data = len(data) - (m-1)*tau
    t_f =150
    #solve for given file
    exponent(tau,m,min_steps, t_0,t_f,delta_t,filename, force_minsteps) 
if __name__ == "__main__":
    main()  