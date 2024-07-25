import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
import rosenstein
import kantz

def plot_growth_factors(times, lyap_exponents,fn,type,t_0,t_f):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log mean divergence")
    plt.title(f"Mean Divergence vs Time for {type}")
    plt.savefig(fg.filename_store_data(type)+"lyapunov_plot.png")
    plt.clf()
    plt.close()
def plot_exponents(types,exponents):
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    ax.set_xticklabels(types)
    ax.scatter(range(len(exponents)),exponents)
    ax.set_xlabel("Type")
    ax.set_ylabel("Exponent")
    ax.set_title("Lyapunov Exponents for Z-Acceleration on Flat Terrain")
    fig.savefig("3_Results/exponents.png")


def welch_method(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency



def exponent(tau,m,min_steps,epsilon,plotting_0,plotting_final,
             delta_t, force_minsteps,types,exponents,type):
    #load data and format
    filename = fg.filename_clean_data(type)
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
    t_f = min_steps*2

    times, data = rosenstein.lyapunov(data,tau,m,min_steps,plotting_0,plotting_final,delta_t)

    # #calculate lyapunov exponents with kantz method
    # times, data = kantz.lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
    #plot growth
    coef=np.polyfit(times[t_0:t_f],data[t_0:t_f],1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,type,t_0,t_f)

    #track exponents and frequencies
    exponents.append(coef[0])
    types.append(type)

    #store times and data in csv
    data = pd.DataFrame(np.column_stack((times,data)),columns=['times','Mean Divergence'])
    data.to_csv(fg.filename_lyapunov(type),index=True)