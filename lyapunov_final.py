import numpy as np
import pandas as pd
from scipy.signal import welch

import filename_generation as fg  # Importing a custom module for filename generation
import rosenstein  # Importing a custom module for Lyapunov calculations
import kantz  # Importing a custom module for Lyapunov calculations

import matplotlib.pyplot as plt

def set_axis_style(ax, labels):
    """
    Function to set the style of the axis in a plot.
    
    Parameters:
    - ax: The axis object to be styled.
    - labels: The labels for the x-axis.
    """
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def plot_growth_factors(times, lyap_exponents, fn, control_type, disturbance, t_0, t_f, coef):
    """
    Function to plot the growth factors of Lyapunov exponents.
    
    Parameters:
    - times: The time values.
    - lyap_exponents: The Lyapunov exponents.
    - fn: The function for the least squares line.
    - control_type: The type of control.
    - disturbance: The type of disturbance.
    - t_0: The starting index for plotting.
    - t_f: The ending index for plotting.
    - coef: The coefficients of the least squares line.
    """
    ax = plt.plot(times, lyap_exponents, label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], fn(times[t_0:t_f]), label=f"Least Squares Line (slope={np.round(coef[0],3)})", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log mean divergence")
    plt.title(f"Log mean divergence vs time for {control_type} Control on {disturbance} Terrain")
    plt.savefig(fg.store_clean_data(disturbance, control_type) + "lyapunov_plot.png")
    plt.clf()
    plt.close()

def plot_exponents(predefined_exponents, neural_exponents, disturbances):
    """
    Function to plot the Lyapunov exponents for predefined and neural control.
    
    Parameters:
    - predefined_exponents: The Lyapunov exponents for predefined control.
    - neural_exponents: The Lyapunov exponents for neural control.
    - disturbances: The types of disturbances.
    """
    plt.scatter(disturbances, predefined_exponents, label="Predefined Control", color="blue")
    plt.scatter(disturbances, neural_exponents, label="Neural Control", color="red")
    plt.xlabel("Disturbance")
    plt.ylabel("Lyapunov Exponent")
    plt.title("Lyapunov Exponents for Neural and Predefined Control")
    plt.legend()
    plt.savefig("3_results/lyapunov_exponents.png")
    plt.clf()
    plt.close()

def welch_method(data):
    """
    Function to calculate the Welch method for spectral density estimation.
    
    Parameters:
    - data: The input data.
    
    Returns:
    - The mean period calculated using the Welch method.
    """
    data = np.reshape(data, (1, -1))
    time_series = data[0]
    f, Pxx = welch(time_series)
    w = Pxx / np.sum(Pxx)
    mean_frequency = np.average(f, weights=w)
    return 1 / mean_frequency

def exponent(tau, m, min_steps, epsilon, plotting_0, plotting_final, delta_t, force_minsteps, predefined_exponents, neural_exponents, disturbance, control_type):
    """
    Function to calculate the Lyapunov exponents.
    
    Parameters:
    - tau: The time delay.
    - m: The embedding dimension.
    - min_steps: The minimum number of steps.
    - epsilon: The threshold for divergence.
    - plotting_0: The starting index for plotting.
    - plotting_final: The ending index for plotting.
    - delta_t: The time step.
    - force_minsteps: Flag to force the minimum number of steps.
    - predefined_exponents: List to store the Lyapunov exponents for predefined control.
    - neural_exponents: List to store the Lyapunov exponents for neural control.
    - disturbance: The type of disturbance.
    - control_type: The type of control.
    """
    filename = fg.filename_clean(disturbance, control_type)  # Generate the filename for clean data
    df = pd.read_csv(filename)  # Read the data from the file
    pdata = df[['roll', 'pitch', 'yaw']]  # Select the relevant columns
    data = pdata.values  # Convert the data to a numpy array

    if not force_minsteps:
        min_steps = welch_method(data)  # Calculate the minimum number of steps using the Welch method
    min_steps = int(min_steps) + 1 if min_steps % 1 != 0 else int(min_steps)  # Round up the minimum number of steps
    t_0 = 0  # Set the starting index for plotting
    t_f = min_steps   # Set the ending index for plotting
    plotting_final = 3 * min_steps  # Set the final index for plotting

    times, data = rosenstein.lyapunov(data, tau, m, min_steps, plotting_0, plotting_final, delta_t)  # Calculate the Lyapunov exponents

    coef = np.polyfit(times[t_0:t_f], data[t_0:t_f], 1)  # Fit a least squares line to the Lyapunov exponents
    poly1d_fn = np.poly1d(coef)  # Create a function for the least squares line
    plot_growth_factors(times, data, poly1d_fn, control_type, disturbance, t_0, t_f, coef)  # Plot the growth factors

    if control_type == 'Predefined':
        predefined_exponents.append(coef[0])  # Store the Lyapunov exponent for predefined control
    else:
        neural_exponents.append(coef[0])  # Store the Lyapunov exponent for neural control

    data = pd.DataFrame(np.column_stack((times, data)), columns=['times', 'Mean Divergence'])  # Create a DataFrame with the times and mean divergence
    data.to_csv(fg.store_clean_data(disturbance, control_type) + 'lyapunovdata.csv', index=True)  # Save the DataFrame to a CSV file
