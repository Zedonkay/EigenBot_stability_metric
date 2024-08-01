import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
import rosenstein
import kantz

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
def plot_growth_factors(times, lyap_exponents,fn,control_type,disturbance,t_0,t_f,coef):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]),label=f"Least Squares Line (slope={np.round(coef[0],3)})", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log mean divergence")
   
    plt.title(f"Log mean divergence vs time for {control_type} Control on {disturbance} Terrain")
    
    plt.savefig(fg.store_clean_data(disturbance,control_type)+"lyapunov_plot.png")
    plt.clf()
    plt.close()
    
def plot_exponents(predefined_exponents,neural_exponents,disturbances):
    ay = plt.scatter(disturbances,predefined_exponents,label="Predefined Control",color="blue")
    plt.scatter(disturbances,neural_exponents,label="Neural Control",color="red")
    plt.xlabel("Disturbance")
    plt.ylabel("Lyapunov Exponent")
    plt.title("Lyapunov Exponents for Neural and Predefined Control")
    plt.legend()
    plt.savefig("3_results/lyapunov_exponents.png")
    plt.clf()
    plt.close()

def welch_method(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency



def exponent(tau,m,min_steps,epsilon,plotting_0,plotting_final,delta_t, 
             force_minsteps,predefined_exponents,neural_exponents,disturbance,control_type):
    #load data and format
    filename = fg.filename_clean(disturbance,control_type)
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
    plotting_final = 3*min_steps

    times, data = rosenstein.lyapunov(data,tau,m,min_steps,plotting_0,plotting_final,delta_t)

    # #calculate lyapunov exponents with kantz method
    # times, data = kantz.lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
    #plot growth
    coef=np.polyfit(times[t_0:t_f],data[t_0:t_f],1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,disturbance,t_0,t_f,coef)

    #track exponents and frequencies
    if(control_type == 'Predefined'):
        predefined_exponents.append(coef[0])
    else:
        neural_exponents.append(coef[0])
    #store times and data in csv
    data = pd.DataFrame(np.column_stack((times,data)),columns=['times','Mean Divergence'])
    data.to_csv(fg.store_clean_data(disturbance,control_type)+'lyapunovdata.csv',index=True)