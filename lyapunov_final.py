import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
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
            mean_distance += np.log(np.linalg.norm(reconstructed_data[k+i]-reconstructed_data[j+i]))
        mean_distance.append(dist/len(vector_addresses[j]) if len(vector_addresses[j])>0 else 0)
    return np.mean(mean_distance)
def kantz_distance(vector_addresses,reconstructed_data, t_0, t_f, delta_t):
    mean_distance = []
    times = []
    for i in range(t_0,t_f):
        mean_distance.append(kantz_mean_distance(vector_addresses,reconstructed_data,i))
        times.append(i*delta_t)
    return times, mean_distance
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

def plot_growth_factors(times, lyap_exponents,fn,control_type,frequency,test):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times, fn(times),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.savefig(fg.store_raw_retest(frequency,test,control_type) + "lyapunov_plot.png")
    plt.clf()
def plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents):
    ay = plt.scatter(centralised_frequencies,centralised_exponents,label="Centralised")
    plt.scatter(distributed_frequencies,distributed_exponents,label="Distributed")
    plt.xlabel("Frequency")
    plt.ylabel("Lyapunov Exponent")
    plt.legend()
    plt.show()
    plt.clf()
def exponent(tau,m,min_steps, t_0,t_f,delta_t, force_minsteps, frequencies,exponents,frequency,control_type,test):
    #load data and format
    filename = fg.filename_raw_retest(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values

    #calculate lyapunov exponents with rosenstein method and plot growth
    times, data = rosenstein_lyapunov(data,tau,m,min_steps,t_0,t_f,delta_t,force_minsteps)
    coef=np.polyfit(times,data,1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,frequency,test)

    #track exponents and frequencies
    exponents.append(coef[0])
    frequencies.append(frequency)

    #store times and data in csv
    data = pd.DataFrame(data,index=times,columns=['lyapunov_exponent'])
    data.to_csv(fg.store_raw_retest(frequency,test,control_type)+"_lyapunovdata.csv",index=True)

def main():
    #parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    t_0 =80
    t_f =150


    #track centralised exponents
    centralised_exponents = []
    centralised_frequencies = []

    #track distributed exponents 
    distributed_exponents = []
    distributed_frequencies = []

    #calculate exponents for centralised control
    print("Calculating exponents for centralised control")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,37,"centralised","1")
    print("done for 37Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,40,"centralised","1")
    print("done for 40Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,60,"centralised","1")
    print("done for 60Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,100,"centralised","1")
    print("done for 100Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,140,"centralised","1")
    print("done for 140Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,180,"centralised","1")
    print("done for 180Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,220,"centralised","1")
    print("done for 220Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,260,"centralised","1")
    print("done for 260Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,300,"centralised","1")
    print("done for 300Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,330,"centralised","1")
    print("done for 330Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,350,"centralised","1")   
    print("done for 350Hz") 
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,400,"centralised","1")
    print("done for 400Hz")

    #calculate exponents for distributed control
    print("Calculating exponents for distributed control")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,500,"distributed","1")
    print("done for 500Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,600,"distributed","1")
    print("done for 600Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,700,"distributed","1")
    print("done for 700Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,800,"distributed","1")
    print("done for 800Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1000,"distributed","1")
    print("done for 1000Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1200,"distributed","1")
    print("done for 1200Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1400,"distributed","1")
    print("done for 1400Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1440,"distributed","1")
    print("done for 1440Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1480,"distributed","1")
    print("done for 1480Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1600,"distributed","1")
    print("done for 1600Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,1800,"distributed","1")
    print("done for 1800Hz")
    exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,2000,"distributed","1")
    print("done for 2000Hz")
    
    #plot exponents
    plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents)

    #store exponents and frequencies in csv
    data = pd.DataFrame(centralised_exponents,index=centralised_frequencies,columns=['lyapunov_exponent'])
    data.to_csv("6_Results/raw_data/centralised/retest/centralised_exponents.csv",index=True)
    data = pd.DataFrame(distributed_exponents,index=distributed_frequencies,columns=['lyapunov_exponent'])
    data.to_csv("6_Results/raw_data/distributed/retest/distributed_exponents.csv",index=True)
    
if __name__ == "__main__":
    main()  