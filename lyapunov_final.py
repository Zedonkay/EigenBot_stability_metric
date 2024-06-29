import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
import rosenstein
import kantz

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

def exponent(tau,m,min_steps,epsilon, g_0,g_f,t_0,t_f,delta_t, force_minsteps,centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents,frequency,control_type,test):
    #load data and format
    filename = fg.filename_clean(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values

    #calculate lyapunov exponents with rosenstein method 
    times, data = rosenstein.lyapunov(data,tau,m,min_steps,g_0,g_f,delta_t,force_minsteps)

    # #calculate lyapunov exponents with kantz method
    # times, data = kantz.lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
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