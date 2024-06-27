import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
import kantz 
import rosenstein

def plot_growth_factors(times, lyap_exponents,fn,control_type,frequency,test):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[80:], fn(times[80:]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.savefig("lyapunov_plot.png")
    plt.clf()
    plt.close()

def plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents):
    ay = plt.scatter(centralised_frequencies,centralised_exponents,label="Centralised")
    plt.scatter(distributed_frequencies,distributed_exponents,label="Distributed")
    plt.xlabel("Frequency")
    plt.ylabel("Lyapunov Exponent")
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

def exponent(tau,m,min_steps,epsilon, t_0,t_f,delta_t, force_minsteps, frequencies,exponents,frequency,control_type,test,filename):
    #load data and format
    #filename = fg.filename_clean(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data=pdata.values

    # #calculate lyapunov exponents with rosenstein method 
    # times, data = rosenstein.lyapunov(data,tau,m,min_steps,t_0,t_f,delta_t,force_minsteps)

    #calculate lyapunov exponents with kantz method
    times, data = kantz.lyapunov(data,tau,m,t_0,t_f,delta_t,epsilon)
    
    #plot growth
    coef=np.polyfit(times[80:],data[80:],1)
    poly1d_fn = np.poly1d(coef)
    plot_growth_factors(times, data,poly1d_fn,control_type,frequency,test)

    #track exponents and frequencies
    exponents.append(coef[0])
    frequencies.append(frequency)

    #store times and data in csv
    data = pd.DataFrame(data,index=times,columns=['average_divergence'])
    data.to_csv("lorentz_lyapunov.csv",index=True)

def main():
    #parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    t_0 =0
    t_f =200
    epsilon = 10


    #track centralised exponents
    centralised_exponents = []
    centralised_frequencies = []

    #track distributed exponents 
    distributed_exponents = []
    distributed_frequencies = []

    exponent(tau,m,min_steps,epsilon,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,100,'centralised','1',"lorentz_data.csv")

    # #calculate exponents for centralised control

    # centralised = [[100,'1'],[100,'2'],[140,'1'],[180,'1'],
    #                [180,'2'],[180,'3'],[220,'1'],[220,'2'],
    #                [220,'3'],[260,'1'],[260,'2'],[260,'3'],
    #                [260,'4'],[300,'1'],[320,'1'],[320,'2'],
    #                [330,'1'],[350,'1'],[350,'2'],[350,'3'],
    #                [370,'1'],[370,'2'],[370,'3']]
   
    # distributed =[[100,'1'],[100,'2'],[100,'3'],[140,'1'],[140,'2'],
    #               [180,'1'],[180,'2'],[220,'1'],[220,'2'],
    #               [260,'1'],[260,'2'],[260,'3'],[300,'1'],[300,'2'],[300,'3'],
    #               [350,'1'],[350,'2'],[370,'1'],[370,'2'],[400,'1'],
    #               [400,'2'],[500,'1'],[600,'1'],[700,'1'],[800,'1'],
    #               [1000,'1'],[1040,'1'],[1120,'1'],[1160,'1'],[1200,'1'],
    #               [1440,'1'],[1600,'1'], [1800,'1'],[2000,'1']]
    

    # print("Calculating exponents for centralised control")
    # for i in centralised:
    #     exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,centralised_frequencies,centralised_exponents,i[0],'centralised',i[1])
    #     print(f"Exponent for centralised {i[0]}Hz test {i[1]} calculated")
    
    # #calculate exponents for distributed control
    # print("Calculating exponents for distributed control")
    # for i in distributed:
    #     exponent(tau,m,min_steps,t_0,t_f,delta_t,force_minsteps,distributed_frequencies,distributed_exponents,i[0],'distributed',i[1])
    #     print(f"Exponent for distributed {i[0]}Hz test {i[1]} calculated")
    
    # #plot exponents
    # plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents)

    # #store exponents and frequencies in csv
    # data = pd.DataFrame(centralised_exponents,index=centralised_frequencies,columns=['mean divergence'])
    # data.to_csv("6_Results/clean_data/centralised/centralised_exponents.csv",index=True)
    # data = pd.DataFrame(distributed_exponents,index=distributed_frequencies,columns=['mean divergence'])
    # data.to_csv("6_Results/clean_data/distributed/distributed_exponents.csv",index=True)
    
if __name__ == "__main__":
    main()  