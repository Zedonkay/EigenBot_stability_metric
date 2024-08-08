import numpy as np
import pandas as pd
import filename_generation as fg
from scipy.signal import welch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

labels = []  # List to store labels for the violin plot

def add_label(violin, label):
    """
    Function to add a label to the violin plot.
    Args:
        violin (matplotlib.container.ViolinPlot): The violin plot object.
        label (str): The label to be added.
    """
    color = violin["bodies"][0].get_facecolor().flatten()  # Get the color of the violin plot
    labels.append((mpatches.Patch(color=color), label))  # Append the color and label to the labels list

def set_axis_style(ax, labels):
    """
    Function to set the style of the x-axis.
    Args:
        ax (matplotlib.axes.Axes): The axes object.
        labels (list): List of labels for the x-axis.
    """
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)  # Set the tick positions and labels for the x-axis
    ax.set_xlim(0.25, len(labels) + 0.75)  # Set the limits of the x-axis

def plot_psd(psds_flat, psds_hill, body_trial, leg_trial):
    """
    Function to plot the PSD*Freq for Z-Acceleration vs Frequency.
    Args:
        psds_flat (list): List of PSD*Freq values for flat terrain.
        psds_hill (list): List of PSD*Freq values for hilly terrain.
        tests (list): List of disturbance labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Create a figure and axes object
    
    # Add violin plots for flat and hilly terrain
    add_label(ax.violinplot(psds_flat, side='high', showmeans=False, showmedians=False, showextrema=False), "Flat")
    add_label(ax.violinplot(psds_hill, side='low', showmeans=False, showmedians=False, showextrema=False), "Hill")
    
    ax.legend(*zip(*labels), loc=9)  # Add legend using the labels list
    ax.set_xlabel("Disturbance")  # Set x-axis label
    ax.set_ylabel("PSD*Freq")  # Set y-axis label
    if(len(body_trial) > len(leg_trial)):
        tests = body_trial
    else:
        tests = leg_trial
    set_axis_style(ax, tests)  # Set style for x-axis
    ax.set_title("PSD*Freq for Z-Acceleration vs Frequency")  # Set title for the plot
    
    fig.savefig("3_results/psd.png")  # Save the figure as an image
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure

def calc(data):
    """
    Function to calculate the PSD using Welch's method.
    Args:
        data (numpy.ndarray): Input data array.
    Returns:
        f (numpy.ndarray): Array of frequencies.
        Pxx (numpy.ndarray): Power spectral density.
    """
    data = np.reshape(data, (1, -1))  # Reshape the data array
    time_series = data[0]  # Extract the time series data
    f, Pxx = welch(time_series)  # Calculate the PSD using Welch's method
    return f, Pxx  # Return the frequencies and PSD

def main(psds_flat, psds_hill, terrain, object, test):
    """
    Main function to process the data and generate the PSD plot.
    Args:
        psds_flat (list): List to store PSD*Freq values for flat terrain.
        psds_hill (list): List to store PSD*Freq values for hilly terrain.
        terrain (str): Terrain type (either "flat" or "hill").
        object (str): Object label.
        test (str): Test label.
    """
    filename = fg.filename_clean(terrain, object, test)  # Generate the filename based on terrain, object, and test
    df = pd.read_csv(filename)  # Read the data from the CSV file
    pdata = df[['az']]  # Extract the 'az' column from the DataFrame
    data = pdata.values  # Convert the DataFrame to a numpy array
    f, Pxx = calc(data)  # Calculate the PSD using Welch's method
    
    if terrain == "flat":
        psds_flat.append(Pxx*f)  # Append the PSD*Freq value to the list
    else:
        psds_hill.append(Pxx*f)  # Append the PSD*Freq value to the list
