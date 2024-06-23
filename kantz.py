import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def kantz_lyapunov(reconstructed_data,epsilon):
    # kantz method for lyapunov exponents
    # not working
    plot_data = []
    for i in range(len(reconstructed_data)):
        print(i)
        epsilon_space=[]
        vector_locations = []
        epsilon_space_next = []
        for j in range(len(reconstructed_data)):
            if i != j:
                dist = np.linalg.norm(reconstructed_data[i]-reconstructed_data[j])
                if dist<epsilon:
                    epsilon_space.append(dist)
                    vector_locations.append(j)
        if i!=len(reconstructed_data)-1:
            for j in vector_locations:
                if(j!=len(reconstructed_data)-1):
                    epsilon_space_next.append(np.linalg.norm(reconstructed_data[i+1]-reconstructed_data[j+1]))
                else:
                    k = vector_locations.index(j)
                    epsilon_space.pop(k)
                    vector_locations.pop(k)

            for j in range(len(epsilon_space)):
                if epsilon_space[j]!=0:
                    epsilon_space[j] = np.log(epsilon_space_next[j]/epsilon_space[j])
            plot_data.append([np.mean(epsilon_space)])
    plot_data = np.array(plot_data)
    plot_data = plot_data[~np.isnan(plot_data)]
    plot_data = np.reshape(plot_data,(1,-1))[0]
    
    return plot_data

