import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.fftpack import fft
import scipy.fft
from scipy.signal import welch
from scipy.signal import find_peaks
import os
import natsort
import tkinter as tk
import ipdb
from matplotlib.widgets import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


inspection_var = 'az'
terrain_type = 'Flat Terrain'
type_left = 'Centralised Controller'
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
                psd_tuples.append(psd*freq)
                psd_labels.append(f'{file_names[j]}')
                
        axs[i].boxplot(psd_tuples)
        axs[i].set_xticklabels(psd_labels, rotation=45, ha='right')
        axs[i].set_xlabel('Update Rate [Hz]')
        axs[i].set_ylabel('Power Spectrum Density * Freq [W*Hz]')
        axs[i].set_ylim([0, 2])
        
    axs[0].set_title(type_left)
    axs[1].set_title(type_right)
    title = f'PSD*Freq Distribution for Z-Acceleration vs. Update Rate on {terrain_type}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)

def plot_box_fft(collection_dfs, columns, collection_fs, collection_filenames):
    fig, axs = plt.subplots(1, len(collection_dfs), figsize=(10, 6))
    
    for i, dfs in enumerate(collection_dfs):
        fs_list = collection_fs[i]
        file_names = collection_filenames[i]
        fft_tuples = []
        fft_labels = []
        
        for j, df in enumerate(dfs):
            for k, col in enumerate(columns):
                yf = np.fft.fft(df[col])
                #xf = np.fft.fftfreq(df['trunc_local_time'].size, df['trunc_local_time'][1] - df['trunc_local_time'][0])
                # Only take the first half of the yf array
                half_length = len(yf) // 2
                yf = yf[:half_length]
                fft_tuples.append(yf)
                #fft_tuples.append(np.abs(yf)) #only plot positive frequencies
                fft_labels.append(f'{file_names[j]}')
                
        #ipdb.set_trace()
        axs[i].boxplot(fft_tuples)
        axs[i].set_xticklabels(fft_labels, rotation=45, ha='right')
        axs[i].set_xlabel('Update Rate [Hz]')
        axs[i].set_ylabel('Fast Fourier Transform (FFT)')
        axs[i].set_ylim([0, 50])
    axs[0].set_title(type_left)
    axs[1].set_title(type_right)
    nyquist_freq = int((1/(df['trunc_local_time'][1] - df['trunc_local_time'][0]))/2)
    title = f'FFT Distribution for Z-Acceleration vs. Update Rate on {terrain_type}\n Nyquist Frequency: {nyquist_freq} Hz'
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
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol =1)
    axs[0].set_title('Centralized')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel(input_variable[0])

    # Plot decentralized_dfs
    for i, df in enumerate(collection_dfs[1]):
        line, = axs[1].plot(df['trunc_local_time'], df[input_variable], label=collection_filenames[1][i])
        cursor = mplcursors.cursor(line)
        cursor.connect("add", lambda sel, i=i: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[1][i] + ": " + str(sel.target[1]))))
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol = 1)
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

# def plot_individual_time_series(input_variable, collection_filenames, collection_dfs):
#     for collection_index, dfs in enumerate(collection_dfs):
#         fig, axs = plt.subplots(len(dfs), 1, sharex=True)
#         annotations = []

#         for df_index, df in enumerate(dfs):
#             line, = axs[df_index].plot(df['trunc_local_time'], df[input_variable], label=collection_filenames[collection_index][df_index])
#             cursor = mplcursors.cursor(line)
#             cursor.connect("add", lambda sel, i=df_index: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[collection_index][i] + ": " + str(sel.target[1]))))
#             axs[df_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol =1)
#             axs[df_index].set_xlabel('Time [s]')
#             #axs[df_index].set_ylabel(input_variable[0])
#             control_type = 'Centralised' if collection_index == 0 else 'Distributed'
#             title = f'{control_type} Z-Acceleration vs. Time on {terrain_type}'
#             plt.suptitle(title)

#         # Add a button for clearing the cursors
#         button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
#         button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
#         def clear_cursors(event):
#             for annotation in annotations:
#                 annotation.set_visible(False)
#             annotations.clear()
#             fig.canvas.draw()
#         button.on_clicked(clear_cursors)

#         plt.tight_layout()
#         plt.show(block=False)

def plot_individual_time_series(input_variable, collection_filenames, collection_dfs):
    for collection_index, dfs in enumerate(collection_dfs):
        fig, axs = plt.subplots(len(dfs), 1, sharex=True)
        annotations = []

        for df_index, df in enumerate(dfs):
            line, = axs[df_index].plot(df['trunc_local_time'], df[input_variable], label=collection_filenames[collection_index][df_index])
            cursor = mplcursors.cursor(line)
            cursor.connect("add", lambda sel, i=df_index: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[collection_index][i] + ": " + str(sel.target[1]))))
            axs[df_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':10}, ncol =1)
            axs[df_index].set_xlabel('Time [s]')
            #axs[df_index].set_ylabel(input_variable[0])
            control_type = 'Centralised' if collection_index == 0 else 'Distributed'
            title = f'{control_type} Z-Acceleration vs. Time on {terrain_type}'
            plt.suptitle(title)
            axs[df_index].set_ylim([-7, 7])
            # Set the title for each subplot to be the corresponding file name
            # axs[df_index].set_title(collection_filenames[collection_index][df_index])
        ylabel = f'Z-Acceleration [m/s^2]'
        # Add a single y-label for all subplots
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
        # Add a button for clearing the cursors
        button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
        def clear_cursors(event):
            for annotation in annotations:
                annotation.set_visible(False)
            annotations.clear()
            fig.canvas.draw()
        button.on_clicked(clear_cursors)

        #plt.tight_layout()
        plt.show(block=False)

def plot_individual_psd(input_variable, collection_filenames, collection_dfs):
    for collection_index, dfs in enumerate(collection_dfs):
        fig, axs = plt.subplots(len(dfs), 1, sharex=True)
        annotations = []
        psd_tuples = []
        psd_labels = []
        file_names = collection_filenames[collection_index]
        for df_index, df in enumerate(dfs):
            for j, df in enumerate(dfs):
                for k, col in enumerate(input_variable):
                    freq, psd = welch(df[col], 100)
                    psd_tuples.append(psd)
                    psd_labels.append(f'{file_names[j]}')
            line, = axs[df_index].plot(freq, psd, label=collection_filenames[collection_index][df_index])
            cursor = mplcursors.cursor(line)
            cursor.connect("add", lambda sel, i=df_index: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[collection_index][i] + ": " + str(sel.target[1]))))
            axs[df_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':10}, ncol =1)
            axs[df_index].set_xlabel('Frequency [Hz]')
            #axs[df_index].set_ylabel('PSD*Freq [W*Hz]')
            control_type = 'Centralised' if collection_index == 0 else 'Distributed'
            title = f'{control_type} Z-Acceleration PSD*Freq vs. Update Rate on {terrain_type}\nNyquist Frequency: 50 Hz'
            plt.suptitle(title)
        ylabel = f'PSD*Freq of Z-Acceleration [W*Hz]'
        # Add a single y-label for all subplots
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')


        # Add a button for clearing the cursors
        button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
        def clear_cursors(event):
            for annotation in annotations:
                annotation.set_visible(False)
            annotations.clear()
            fig.canvas.draw()
        button.on_clicked(clear_cursors)

        # plt.tight_layout()
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
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol = 1)
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

def plot_individual_FFT(input_variable, collection_filenames, collection_dfs):
    for collection_index, dfs in enumerate(collection_dfs):
        fig, axs = plt.subplots(len(dfs), 1, sharex=True)
        annotations = []

        for df_index, df in enumerate(dfs):
            # yf = np.fft.fft(np.array(df[input_variable]))
            # xf = np.fft.fftfreq(df['trunc_local_time'].size, .01)
            # xf = np.fft.fftfreq(df['trunc_local_time'].size, df['trunc_local_time'][1] - df['trunc_local_time'][0])
            #ipdb.set_trace()
            # # Only take the first half of the xf and yf arrays
            # half_length = len(xf) // 2
            # xf = xf[:half_length]
            # yf = yf[:half_length]
            yf = scipy.fft.fft(df[input_variable])
            xf = scipy.fft.fftfreq(len(df['trunc_local_time']), .01)
            line, = axs[df_index].plot(xf, yf.real, label=collection_filenames[1][df_index])
            # cursor = mplcursors.cursor(line)
            # cursor.connect("add", lambda sel, i=df_index: (annotations.append(sel.annotation), sel.annotation.set_text(collection_filenames[1][df_index] + ": " + str(sel.target[1]))))
            
            axs[df_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':6}, ncol =1)
            axs[df_index].set_xlabel('Frequency [Hz]')
            axs[df_index].set_ylabel(input_variable[0]+ ' Magnitude')
            control_type = 'Centralized' if collection_index == 0 else 'Distributed'
            nyquist_freq = int((1/(df['trunc_local_time'][1] - df['trunc_local_time'][0]))/2)
            title = f'{control_type} {input_variable[0]} FFT vs. Freq. on {terrain_type}\nNyquist Frequency: 50 Hz'
            plt.suptitle(title)

        # Add a button for clearing the cursors
        button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(button_ax, 'Clear Cursors', hovercolor='0.975')
        def clear_cursors(event):
            for annotation in annotations:
                annotation.set_visible(False)
            annotations.clear()
            fig.canvas.draw()
        button.on_clicked(clear_cursors)

        plt.tight_layout()
        plt.show(block=False)


# Define two lookup tables for cutoff values
cutoff_lookup_1 = {
    '140Hz.csv': {'front_cutoff': 6.86, 'end_cutoff': 12.56},  
    '220Hz.csv': {'front_cutoff': 4.87, 'end_cutoff': 10.78},
    '350Hz.csv': {'front_cutoff': 5.83, 'end_cutoff': 11.41},
    '370Hz.csv': {'front_cutoff': 3.70, 'end_cutoff': 9.48},
    '400Hz.csv': {'front_cutoff': 4.83, 'end_cutoff': 10.5}
}

cutoff_lookup_2 = {
    '140Hz.csv': {'front_cutoff': 2.41, 'end_cutoff': 8.39}, 
    '220Hz.csv': {'front_cutoff': 3.2, 'end_cutoff': 9.23},
    '350Hz.csv': {'front_cutoff': 3.33, 'end_cutoff': 9.31},
    '370Hz.csv': {'front_cutoff': 6.65, 'end_cutoff': 12.72},
    '400Hz.csv': {'front_cutoff': 3.96, 'end_cutoff': 9.8},
    '500Hz.csv': {'front_cutoff': 5.9, 'end_cutoff': 11.91},
    '600Hz.csv': {'front_cutoff': 3.62, 'end_cutoff': 9.7},
    '800Hz.csv': {'front_cutoff': 2.62, 'end_cutoff': 8.71},
    '1000Hz.csv': {'front_cutoff': 7.42, 'end_cutoff': 13.74},
    '1120Hz.csv': {'front_cutoff': 4.01, 'end_cutoff': 9.83},
    '1200Hz.csv': {'front_cutoff': 4.4, 'end_cutoff': 10.4},
    '1440Hz.csv': {'front_cutoff': 5.29, 'end_cutoff': 11.33},
    '1800Hz.csv': {'front_cutoff': 5.85, 'end_cutoff': 11.88},
    '2000Hz.csv': {'front_cutoff': 4.56, 'end_cutoff': 10.75}
}

def truncate_data(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return None

    # Get the filename and folder name from the file path
    folder_name = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    #ipdb.set_trace()
    # Choose the lookup table based on the folder name
    if folder_name == 'centralised':
        cutoff_lookup = cutoff_lookup_1
    elif folder_name == 'distributed':
        cutoff_lookup = cutoff_lookup_2
    else:
        print("No cutoff values found for this folder name.")
        return None

    # Get the cutoff values from the lookup table
    if filename in cutoff_lookup:
        front_cutoff = cutoff_lookup[filename]['front_cutoff']
        end_cutoff = cutoff_lookup[filename]['end_cutoff']
    else:
        print("No cutoff values found for this filename: " + filename)
        return None

    # Load data
    df = pd.read_csv(file_path)

    # Create local_time
    df['local_time'] = df['timestamp'] - df['timestamp'][0]

    # Truncate data based on cutoff timestamps
    df = df.loc[(df['local_time'] >= front_cutoff) & (df['local_time'] <= end_cutoff)]
    df['trunc_local_time'] = df['local_time'] - front_cutoff

    # Reset the index
    df.reset_index(drop=True, inplace=True)
    return df

def process_file(file_path):
    # Load data
    df = pd.read_csv(file_path)

    df = truncate_data(file_path)

    if df is None:
        return None
    #ipdb.set_trace()

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
    #plot_time_series([inspection_var], collection_filenames, collection_dfs)
    plot_individual_time_series([inspection_var], collection_filenames, collection_dfs)
    #plot_FFT([inspection_var], collection_filenames, collection_dfs)
    plot_individual_FFT([inspection_var], collection_filenames, collection_dfs)
    plot_individual_psd([inspection_var], collection_filenames, collection_dfs)
    plot_box_psd(collection_dfs, [inspection_var], collection_fs, collection_filenames)
    plot_box_fft(collection_dfs, [inspection_var], collection_fs, collection_filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frequency Metric for Centralized and Decentralized Data.')
    parser.add_argument('--centralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--decentralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--mode', type=str, help='')
    args = parser.parse_args()
    main(args)
    input()
