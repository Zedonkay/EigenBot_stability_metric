import numpy as np
import pandas as pd
import filename_generation as fg
import rosenstein
# import kantz
import matplotlib.pyplot as plt
from scipy.signal import welch

def welch_method(data):
    """
    Apply Welch's method to estimate the mean period.

    Parameters:
    - data: array-like, input data

    Returns:
    - mean period
    """
    # Reshape the data into a 1-dimensional array
    data = np.reshape(data, (1, -1))
    time_series = data[0]
    
    # Apply Welch's method to estimate the power spectral density
    f, Pxx = welch(time_series)
    
    # Calculate the normalized weights
    w = Pxx / np.sum(Pxx)
    
    # Calculate the mean frequency using the weighted average
    mean_frequency = np.average(f, weights=w)
    
    # Return the reciprocal of the mean frequency as the mean period
    return 1 / mean_frequency

def plot_growth_factors(times, lyap_exponents, fn, date, terrain, trial, t_0, t_f,coef):
    """
    Plot the growth factors of Lyapunov exponents over time.

    Parameters:
    - times: array-like, time values
    - lyap_exponents: array-like, Lyapunov exponents
    - fn: function, least squares line function
    - date: str, date of the data
    - terrain: str, type of terrain
    - trial: int, trial number
    - t_0: int, starting index for plotting
    - t_f: int, ending index for plotting
    """
    plt.plot(times, lyap_exponents, label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]), label=f"Least Squares Line (exponent = {coef})", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log mean divergence")
    plt.title(f"Mean Divergence vs time for {date} data on {terrain} terrain (trial {trial})")
    plt.savefig(fg.filename_store_data(date, terrain, trial) + "lyapunov_plot.svg", format="svg")
    plt.clf()
    plt.close()

def plot_exponents(exponents, date, terrain, trial):
    """
    Plot the Lyapunov exponents for Z-Acceleration.

    Parameters:
    - trials: array-like, trial numbers
    - exponents: array-like, Lyapunov exponents
    - date: str, date of the data
    - terrain: str, type of terrain
    - trial: int, trial number
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(list(exponents.keys()), list(exponents.values()), label="Lyapunov Exponents", color="blue")
    
    ax.set_xlabel("Trial")
    ax.set_ylabel("Exponent")
    ax.set_title("Lyapunov Exponents for Z-Acceleration for " + date + " data on " + terrain + " terrain")
    fig.savefig(fg.filename_big(date, terrain, trial) + "exponents.svg", format="svg")
    plt.clf()
    plt.close()



def exponent(tau, m, min_steps, plotting_0, plotting_final,
             delta_t, force_minsteps, exponents, date, terrain, trial,trial_data):
    """
    Calculate Lyapunov exponents.

    Parameters:
    - tau: int, time delay
    - m: int, embedding dimension
    - min_steps: int, minimum number of steps
    - plotting_0: bool, whether to plot initial state
    - plotting_final: bool, whether to plot final state
    - delta_t: float, time step size
    - force_minsteps: bool, whether to force minimum steps calculation
    - trials: array-like, trial numbers
    - exponents: array-like, Lyapunov exponents
    - date: str, date of the data
    - terrain: str, type of terrain
    - trial: int, trial number
    """
    # Read the data from the file
    filename = fg.filename_clean_data(date, terrain, trial)
    df = pd.read_csv(filename)
    pdata = df[['pz']]
    data = pdata.values
     # Calculate the minimum number of steps using Welch's method
    if not force_minsteps:
        min_steps = welch_method(data)
    
    # Round up the minimum number of steps
    if min_steps % 1 != 0:
        min_steps = int(min_steps) + 1
    else:
        min_steps = int(min_steps)
    
    # Set the initial and final time indices for plotting
    t_0 = 0
    t_f = min_steps
    plotting_final = min_steps * 2
    
    # Calculate the Lyapunov exponents using Rosenstein's algorithm
    times, data = rosenstein.lyapunov(data, tau, m, min_steps, plotting_0, plotting_final, delta_t)
    # Fit a least squares line to the Lyapunov exponents
    coef = np.polyfit(times[t_0:t_f], data[t_0:t_f], 1)
    poly1d_fn = np.poly1d(coef)
    
    # Plot the growth factors of Lyapunov exponents over time
    plot_growth_factors(times, data, poly1d_fn, date, terrain, trial, t_0, t_f,coef[0])
    
    # Append the Lyapunov exponent and trial number to the respective lists
    exponents.update({trial: coef[0]})
    trial_data.append(coef[0])
    
    # Save the Lyapunov exponents data to a CSV file
    data = pd.DataFrame(np.column_stack((times, data)), columns=['times', 'Mean Divergence'])
    data.to_csv(fg.filename_lyapunov(date, terrain, trial), index=True)
