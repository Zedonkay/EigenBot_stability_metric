import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg


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

def plot_growth_factors(times, lyap_exponents,fn,control_type,frequency,test,t_0,t_f):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.savefig(fg.store_clean_data(frequency,test,control_type)+"lyapunov_plot.png")
    plt.clf()
    plt.close()
def plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents):
    ay = plt.scatter(centralised_frequencies,centralised_exponents,label="Centralised")
    plt.scatter(distributed_frequencies,distributed_exponents,label="Distributed")
    plt.xlabel("Frequency")
    plt.ylabel("Lyapunov Exponent")
    plt.legend()
    plt.savefig("6_Results/clean_data/lyapunov_exponents.png")
    plt.clf()
    plt.close()

def kantz_mean_distance(vector_addresses,reconstructed_data,i):
    #calculate mean distance for kantz method
    distances = []
    for j in range(len(reconstructed_data)-i):
        for k in vector_addresses[j]:
            if k+i<len(reconstructed_data):
                distances.append(np.log(np.linalg.norm(reconstructed_data[k+i]-reconstructed_data[j+i])))

    return np.mean(distances)
def kantz_distance(vector_addresses,reconstructed_data, t_0, t_f, delta_t):
    mean_distance = []
    times = []
    for i in range(t_0,t_f):
        if(i%5==0):
            print(i)
        mean_distance.append(kantz_mean_distance(vector_addresses,reconstructed_data,i))
        times.append(i*delta_t)
    return times, mean_distance
def find_epsilon_vectors(reconstructed_data,epsilon):
    vector_addresses=[]
    for i in range(len(reconstructed_data)): 
        if(i%100==0):
            print("on vector ",i)
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
    print("data reconstructed ")
    #find vectors within epsilon distance of each vector
    vector_addresses = find_epsilon_vectors(reconstructed_data,epsilon)
    print("vectors found")
    #calculate log distance
    times, mean_distances = kantz_distance(vector_addresses,reconstructed_data,t_0,t_f,delta_t)
    mean_distances = np.array(mean_distances)
    times = np.array(times)
        
   
    return times, mean_distances

def exponent(tau,m,min_steps,epsilon, g_0,g_f,t_0,t_f,delta_t, force_minsteps,centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents,frequency,control_type,test):
    #load data and format
    filename = fg.filename_clean(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values

    #calculate lyapunov exponents with rosenstein method 
    times, data = rosenstein_lyapunov(data,tau,m,min_steps,g_0,g_f,delta_t,force_minsteps)

    # #calculate lyapunov exponents with kantz method
    # times, data = kantz_lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
    #plot growth
    coef=np.polyfit(times[t_0:t_f],data[t_0:t_f],1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,frequency,test,t_0,t_f)

    #track exponents and frequencies
    if(control_type == 'centralised'):
        centralised_exponents.append(coef[0])
        centralised_frequencies.append(frequency)
    else:
        distributed_exponents.append(coef[0])
        distributed_frequencies.append(frequency)
    #store times and data in csv
    data = pd.DataFrame(np.column_stack((times,data)),columns=['times','average_divergence'])
    data.to_csv(fg.store_clean_data(frequency,test,control_type)+'lyapunovdata.csv',index=True)

def main():
    #parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    t_0 =0
    t_f =300
    epsilon = 10


    #track centralised exponents
    centralised_exponents = []
    centralised_frequencies = []

    #track distributed exponents 
    distributed_exponents = []
    distributed_frequencies = []

    

    #calculate exponents for centralised control

 

    print("Calculating exponents for centralised control")
    for i in centralised:
        exponent(tau,m,min_steps,epsilon,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents,i[0],'centralised',i[1])
        print(f"Exponent for centralised {i[0]}Hz test {i[1]} calculated")
    
    #calculate exponents for distributed control
    print("Calculating exponents for distributed control")
    for i in distributed:
        exponent(tau,m,min_steps,epsilon,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents,i[0],'distributed',i[1])
        print(f"Exponent for distributed {i[0]}Hz test {i[1]} calculated")
    
    #plot exponents
    plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents)

    #store exponents and frequencies in csv
    data = pd.DataFrame(np.column_stack((centralised_frequencies,centralised_exponents)),columns=['frequency','exponent'])
    data.to_csv("6_Results/clean_data/centralised/centralised_exponents.csv",index=True)
    data = pd.DataFrame(np.column_stack((distributed_frequencies,distributed_exponents)),columns=['frequency','exponent'])
    data.to_csv("6_Results/clean_data/distributed/distributed_exponents.csv",index=True)
    
if __name__ == "__main__":
    main()  