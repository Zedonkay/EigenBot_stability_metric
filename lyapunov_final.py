import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
import rosenstein
import kantz
from scipy.signal import welch

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
    plt.xlabel("Update Rate [Hz]")
    plt.ylabel("Lyapunov Exponent")
    plt.legend()
    plt.savefig("6_Results/clean_data/lyapunov_exponents.png")
    plt.clf()
    plt.close()

def welch_method(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency



def exponent(tau,m,min_steps,epsilon,plotting_0,plotting_final,delta_t, force_minsteps,centralised_exponents,distributed_exponents,frequency,control_type,test):
    #load data and format
    filename = fg.filename_clean(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values
    #calculate lyapunov exponents with rosenstein method 
    if not force_minsteps:
        min_steps = welch_method(data)
    if min_steps%1 != 0:
        min_steps = int(min_steps)+1
    else:
        min_steps = int(min_steps)
    t_0 = 0
    t_f = 2*min_steps
    print(min_steps)

    times, data = rosenstein.lyapunov(data,tau,m,min_steps,plotting_0,plotting_final,delta_t)

    # #calculate lyapunov exponents with kantz method
    # times, data = kantz.lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
    #plot growth
    coef=np.polyfit(times[t_0:t_f],data[t_0:t_f],1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,frequency,test,t_0,t_f)

    #track exponents and frequencies
    if(control_type == 'centralised'):
        centralised_exponents.append(coef[0])
    else:
        distributed_exponents.append(coef[0])
    #store times and data in csv
    data = pd.DataFrame(np.column_stack((times,data)),columns=['times','average_divergence'])
    data.to_csv(fg.store_clean_data(frequency,test,control_type)+'lyapunovdata.csv',index=True)