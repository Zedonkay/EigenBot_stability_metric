import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg

def plot_growth_factors(times, lyap_exponents,fn,type,t_0,t_f):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.savefig(fg.filename_store_data(type)+"lyapunov_plot.png")
    plt.clf()
    plt.close()


def polyfit_csv(times,data, t_0, t_f):
    # Perform polynomial fitting
    coef = np.polyfit(times[t_0:t_f], data[t_0:t_f], 1)
    poly1d_fn = np.poly1d(coef)
    # Return the fitted polynomial function
    return coef[0],poly1d_fn

def main(t_0, t_f,type):
   # Read the CSV file into a DataFrame
    filename = fg.filename_clean_data(type)
    df = pd.read_csv(filename)

    # Extract the times and data from the DataFrame
    times = df['times']
    data = df['average_divergence']

    # Perform polynomial fitting
    exponent,poly1d_fn = polyfit_csv(times, data, t_0, t_f)
    # Plot the growth factors
    plot_growth_factors(times, data, poly1d_fn, type, t_0, t_f)