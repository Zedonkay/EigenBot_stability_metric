import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.signal import find_peaks
import os
import natsort
import tkinter as tk
import ipdb
from matplotlib.widgets import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

inspection_var = 'ax'
terrain_type = 'Flat Terrain'
type_left = 'Centralized Controller'
type_right = 'Distributed Controller'

#plotting
def plot_box_psd(collection_dfs, columns, collection_fs, collection_filenames):
    fig, axs = plt.subplots(1, len(collection_dfs), figsize=(10, 6))
    
    for i, dfs in enumerate(collection_dfs):
        fs_list = collection_fs[i]
        file_names = collection_filenames[i]
        psd_tuples = []
        psd_labels = []
        
        for j, df in enumerate(dfs):
            for k, col in enumerate(columns):
                freq, psd = welch(df[col], fs_list[j])
                psd_tuples.append(psd)
                psd_labels.append(f'{file_names[j]}')
                
        axs[i].boxplot(psd_tuples)
        axs[i].set_xticklabels(psd_labels, rotation=45, ha='right')
        axs[i].set_xlabel('Update Rate [Hz]')
        axs[i].set_ylabel('Power Spectrum Density (PSD) [W]')
        axs[i].set_ylim([0, .1])
        
    axs[0].set_title(type_left)
    axs[1].set_title(type_right)
    title = f'PSD Distribution for {inspection_var} on {terrain_type} over Update Rates'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)

def plot_time_series(input_variable, collection_filenames, collection_dfs):
    fig, axs = plt.subplots(2, 1, sharex=True)

    annotations = []

    # Plot centralized_dfs
    for i, df in enumerate(collection_dfs[0]):
        line, = axs[0].plot(df['trunc_local_time'], df[input_variable], label=collection_filenames[0][i])
        cursor = mplcursors.cursor(line)
        cursor.connect("add", lambda sel, i=i: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[0][i] + ": " + str(sel.target[1]))))
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol =2)
    axs[0].set_title('Centralized')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel(input_variable[0])

    # Plot decentralized_dfs
    for i, df in enumerate(collection_dfs[1]):
        line, = axs[1].plot(df['trunc_local_time'], df[input_variable], label=collection_filenames[1][i])
        cursor = mplcursors.cursor(line)
        cursor.connect("add", lambda sel, i=i: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[1][i] + ": " + str(sel.target[1]))))
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol = 2)
    axs[1].set_title('Decentralized')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(input_variable[0])

    # Add a button for clearing the cursors
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
    def clear_cursors(event):
        for annotation in annotations:
            annotation.set_visible(False)
        annotations.clear()
        fig.canvas.draw()
    button.on_clicked(clear_cursors)
    title = f'{input_variable[0]} vs. Time on {terrain_type}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)

def plot_FFT(input_variable, collection_filenames, collection_dfs):
    fig, axs = plt.subplots(2, 1, sharex=True)

    annotations = []

    # Plot centralized_dfs
    for i, df in enumerate(collection_dfs[0]):
        # Compute FFT
        yf = np.fft.fft(df[input_variable])
        xf = np.fft.fftfreq(df['trunc_local_time'].size, df['trunc_local_time'][1] - df['trunc_local_time'][0])
        
        line, = axs[0].semilogy(xf, np.abs(yf), label=collection_filenames[0][i])
        cursor = mplcursors.cursor(line)
        cursor.connect("add", lambda sel, i=i: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[0][i] + ": " + str(sel.target[1]))))
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol =2)
    axs[0].set_title('Centralized')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_ylabel('FFT of ' + input_variable[0])

    # Plot decentralized_dfs
    for i, df in enumerate(collection_dfs[1]):
        # Compute FFT
        yf = np.fft.fft(df[input_variable])
        xf = np.fft.fftfreq(df['trunc_local_time'].size, df['trunc_local_time'][1] - df['trunc_local_time'][0])
        
        line, = axs[1].semilogy(xf, np.abs(yf), label=collection_filenames[1][i])
        cursor = mplcursors.cursor(line)
        cursor.connect("add", lambda sel, i=i: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[1][i] + ": " + str(sel.target[1]))))
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol = 2)
    axs[1].set_title('Decentralized')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('FFT of ' + input_variable[0])

    # Add a button for clearing the cursors
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
    def clear_cursors(event):
        for annotation in annotations:
            annotation.set_visible(False)
        annotations.clear()
        fig.canvas.draw()
    button.on_clicked(clear_cursors)
    title = f'FFT of {input_variable[0]} vs. Frequency'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)

def process_file(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Create local_time
    df['local_time'] = df['timestamp'] - df['timestamp'][0]

    # Define offsets
    front_offset = 0
    df = df.loc[(df['local_time'] >= front_offset)]
    df['trunc_local_time'] = df['local_time'] - front_offset

    fs = 1.0 / np.mean(np.diff(df['trunc_local_time']))

    # Convert quaternion to Euler angles
    df['roll'] = np.arctan2(2*(df['oy']*df['oz'] + df['ow']*df['ox']), df['ow']*df['ow'] - df['ox']*df['ox'] - df['oy']*df['oy'] + df['oz']*df['oz'])
    df['pitch'] = -np.arcsin(2*(df['ox']*df['oz'] - df['ow']*df['oy']))
    df['yaw'] = np.arctan2(2*(df['ox']*df['oy'] + df['ow']*df['oz']), df['ow']*df['ow'] + df['ox']*df['ox'] - df['oy']*df['oy'] - df['oz']*df['oz'])
    # Compute velocities, accelerations, and angular velocity
    df['vx'] = np.gradient(df['px'], df['trunc_local_time'])
    df['vy'] = np.gradient(df['py'], df['trunc_local_time'])
    df['vz'] = np.gradient(df['pz'], df['trunc_local_time'])
    
    df['ax'] = np.gradient(df['vx'], df['trunc_local_time'])
    df['ay'] = np.gradient(df['vy'], df['trunc_local_time'])
    df['az'] = np.gradient(df['vz'], df['trunc_local_time'])

    df['rollRate'] = np.gradient(df['roll'], df['trunc_local_time'])
    df['pitchRate'] = np.gradient(df['pitch'], df['trunc_local_time'])
    df['yawRate'] = np.gradient(df['yaw'], df['trunc_local_time'])

    #df is a matrix with rows being time and columns being the features
    return df, fs

def main(args):
    centralized_files = glob.glob(args.centralized_file_path)
    decentralized_files = glob.glob(args.decentralized_file_path)

    #sort file_paths in numerical order
    centralized_files = natsort.natsorted(centralized_files)
    decentralized_files = natsort.natsorted(decentralized_files)

    centralized_dfs_fs = [process_file(file) for file in centralized_files] # list of tuples
    decentralized_dfs_fs = [process_file(file) for file in decentralized_files]
    #ipdb.set_trace()
    centralized_dfs, centralized_fs_list = zip(*centralized_dfs_fs) #unpacking the list of tuples
    decentralized_dfs, decentralized_fs_list = zip(*decentralized_dfs_fs)
    #print dimensions of centralized_dfs

    #centralized_dfs is a tuple of dataframes. centralized_dfs[0].shape will return 2228x22 for example

    centralized_file_names = [os.path.splitext(os.path.basename(file))[0] for file in centralized_files]
    decentralized_file_names = [os.path.splitext(os.path.basename(file))[0] for file in decentralized_files]

    collection_dfs = [centralized_dfs, decentralized_dfs]
    collection_fs = [centralized_fs_list, decentralized_fs_list]
    collection_filenames = [centralized_file_names, decentralized_file_names]
    plot_time_series([inspection_var], collection_filenames, collection_dfs)
    plot_FFT([inspection_var], collection_filenames, collection_dfs)
    #plot_box_psd(collection_dfs, [inspection_var], collection_fs, collection_filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frequency Metric for Centralized and Decentralized Data.')
    parser.add_argument('--centralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--decentralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--mode', type=str, help='')
    args = parser.parse_args()
    main(args)
    input()
