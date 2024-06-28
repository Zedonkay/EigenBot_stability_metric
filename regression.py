import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_growth_factors(times, lyap_exponents,fn,control_type,frequency,test,t_0,t_f):
    """Plot Lyapunov exponents."""
    ax = plt.plot(times,lyap_exponents,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Average divergence")
    plt.savefig("lyapunov_plot.png")
    plt.clf()
    plt.close()

def polyfit_csv(times,data, t_0, t_f):
    

    # Perform polynomial fitting
    coef = np.polyfit(times[t_0:t_f], data[t_0:t_f], 1)
    poly1d_fn = np.poly1d(coef)
    print(coef[0])
    # Return the fitted polynomial function
    return poly1d_fn

def main(file_path, t_0, t_f,control_type,frequency,test):
   # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract the times and data from the DataFrame
    times = df['times']
    data = df['average_divergence']

    # Perform polynomial fitting
    poly1d_fn = polyfit_csv(times, data, t_0, t_f)
    # Plot the growth factors
    plot_growth_factors(times, data, poly1d_fn, control_type, frequency, test,t_0,t_f)

if __name__ == "__main__":
    main("lorentz_lyapunov.csv", 25, 180, "centralised", 0.1, "test")