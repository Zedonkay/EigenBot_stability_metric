import numpy as np


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
