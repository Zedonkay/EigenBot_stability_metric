#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

# global variables
# file_paths = [f'test/20240808_164938_terrain/data__dev_ttyACM{i}.npz' for i in range(6)]
file_paths = [f'test/20240808_165323_good_flat_walk/data__dev_ttyACM{i}.npz' for i in range(6)]
leg_names = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
indices = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 15))
forces = []
plot_start_time = 76 #seconds

def butter_lowpass(cutoff, fs, order=5):
    """
    Design a lowpass Butterworth filter.
    
    Parameters:
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter.
    
    Returns:
    b, a (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a lowpass Butterworth filter to the data.
    
    Parameters:
    data (ndarray): The input signal.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency of the signal.
    order (int): The order of the filter.
    
    Returns:
    y (ndarray): The filtered signal.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_norm(forces, offset):
    """
    Calculate the norm (magnitude) of the forces for each leg.
    
    Parameters:
    forces (list): List of force measurements.
    offset (int): Offset to access the correct indices for each leg.
    
    Returns:
    norm_forces (list): List of norm (magnitude) of forces for each leg.
    """
    norm_forces = [
        np.sqrt(forces[0+offset]**2 + forces[1+offset]**2 + forces[2+offset]**2),
        np.sqrt(forces[3+offset]**2 + forces[4+offset]**2 + forces[5+offset]**2),
        np.sqrt(forces[6+offset]**2 + forces[7+offset]**2 + forces[8+offset]**2),
        np.sqrt(forces[9+offset]**2 + forces[10+offset]**2 + forces[11+offset]**2)
    ]
    return norm_forces

def find_peaks_troughs(force_avg):
    """
    Find the peaks and troughs in the average force signal.
    
    Parameters:
    force_avg (ndarray): The average force signal.
    
    Returns:
    peaks (ndarray): Indices of the peaks in the signal.
    troughs (ndarray): Indices of the troughs in the signal.
    """
    peaks, _ = find_peaks(force_avg)
    inverted_force_avg = -force_avg
    troughs, _ = find_peaks(inverted_force_avg)
    return peaks, troughs

def find_swing_stance(force_avg):
    """
    Determine the swing and stance phases based on the average force signal.
    
    Parameters:
    force_avg (ndarray): The average force signal.
    
    Returns:
    stance (ndarray): Boolean array indicating stance phase.
    swing (ndarray): Boolean array indicating swing phase.
    """
    peak_to_peak = np.max(force_avg) - np.min(force_avg)
    threshold = np.min(force_avg) + peak_to_peak / 4
    stance = force_avg >= threshold
    swing = force_avg < threshold
    return stance, swing

def process_file(file_path, indices, leg_name, ax):
    """
    Process a single file to extract and plot force data.
    
    Parameters:
    file_path (str): Path to the data file.
    indices (list): List of indices to extract force data.
    leg_name (str): Name of the leg (e.g., 'L1', 'R1').
    ax (matplotlib.axes.Axes): Matplotlib Axes object for plotting.
    """
    data = np.load(file_path)
    time = data['time'] - data['time'][0]
    sampling_rate = 1 / np.mean(np.diff(time))
    force = [data[f'index_{j}'] for j in indices]
    forces.extend(force)
    hall_effect_sensor_offset = 12 * leg_names.index(leg_name)
    
    norm_forces = find_norm(forces, hall_effect_sensor_offset)
    force_avg = np.mean(norm_forces, axis=0)
    force_avg = lowpass_filter(force_avg, int(sampling_rate/2)-1, int(sampling_rate))
    
    peaks, troughs = find_peaks_troughs(force_avg)
    
    plot_time_offset = np.where(time >= plot_start_time)[0][0]
    plotting_time = time[plot_time_offset:]
    
    ax.plot(plotting_time, force_avg[plot_time_offset:], label='Average Force')
    adjusted_peaks = peaks[peaks >= plot_time_offset] - plot_time_offset
    adjusted_troughs = troughs[troughs >= plot_time_offset] - plot_time_offset
    ax.scatter(plotting_time[adjusted_peaks], force_avg[plot_time_offset:][adjusted_peaks], color='red', label='Peak')
    ax.scatter(plotting_time[adjusted_troughs], force_avg[plot_time_offset:][adjusted_troughs], color='green', label='Trough')
    
    stance, swing = find_swing_stance(force_avg)
    
    ax.fill_between(plotting_time, 0, 1, where=stance[plot_time_offset:], color='darkgray', alpha=0.5, transform=ax.get_xaxis_transform())
    ax.fill_between(plotting_time, 0, 1, where=swing[plot_time_offset:], color='lightgray', alpha=0.5, transform=ax.get_xaxis_transform())
    
    ax.set_title(f'Leg {leg_name}')
    ax.set_ylabel('Force Magnitude')
    ax.legend()
    df = pd.DataFrame({'time': plotting_time, 'force': force_avg[plot_time_offset:]})
    if leg_name[0:1] == 'L':
        df.to_csv(f'1_clean_data/{leg_name}.csv', index=False)
    else:
        df.to_csv(f'1_clean_data/L{int(leg_name[1:])+3}.csv', index=False)


def main():
    """
    Main function to process all files and plot the results.
    """
    for i, file_path in enumerate(file_paths):
        process_file(file_path, indices, leg_names[i], axs[i])
    
    axs[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('force_data.png')

if __name__ == "__main__":
    main()