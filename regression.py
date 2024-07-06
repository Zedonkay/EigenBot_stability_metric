import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg

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

def polyfit_csv(times,data, t_0, t_f):
    

    # Perform polynomial fitting
    coef = np.polyfit(times[t_0:t_f], data[t_0:t_f], 1)
    poly1d_fn = np.poly1d(coef)
    # Return the fitted polynomial function
    return coef[0],poly1d_fn

def main(t_0, t_f,control_type,frequency,test):
   # Read the CSV file into a DataFrame
    filename = fg.filename_lyapunov(frequency, test, control_type)
    df = pd.read_csv(filename)

    # Extract the times and data from the DataFrame
    times = df['times']
    data = df['average_divergence']

    # Perform polynomial fitting
    exponent,poly1d_fn = polyfit_csv(times, data, t_0, t_f)
    # Plot the growth factors
    plot_growth_factors(times, data, poly1d_fn, control_type, frequency, test,t_0,t_f)

    df = pd.read_csv(fg.filename_exponents(frequency, test, control_type))
    df.at[frequency,"mean divergence"] = exponent
    df.to_csv(fg.filename_exponents(frequency, test, control_type),index=False)

    df_cent = pd.read_csv(fg.filename_exponents(100, test, "centralised"))
    df_dist = pd.read_csv(fg.filename_exponents(100, test, "distributed"))

    centralised_frequencies = df_cent['frequency']
    centralised_exponents = df_cent["exponent"]
    distributed_frequencies = df_dist['frequency']
    distributed_exponents = df_dist["exponent"]
    plot_exponents(centralised_frequencies,centralised_exponents,distributed_frequencies,distributed_exponents)
if __name__ == "__main__":
    #main(80, 200, "centralised", 330, "1")
    pass