import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.signal import find_peaks
import os
import natsort
import ipdb

inspection_var_label = 'Acceleration in Y'
inspection_var = 'ay'
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
    title = f'PSD Distribution for {inspection_var_label} on {terrain_type} over Update Rates'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)



# def plot_box_psd(dfs, columns, fs, file_names):
#     fig, axs = plt.subplots(1, number_of_dfs, figsize=(10, 6))
#     psd_tuples = []
#     psd_labels = []
    
#     for i, df in enumerate(dfs):
#         for j, col in enumerate(columns):
#             # ipdb.set_trace()
#             freq, psd = welch(df[col], fs[i])
#             psd_tuples.append(psd)
#             psd_labels.append(f'{col}_{file_names[i]}')
            
#     axs.boxplot(psd_tuples)
#     axs.set_xticklabels(psd_labels, rotation=45, ha='right')
#     axs.set_xlabel('PSD Variable')
#     axs.set_ylabel('PSD Units')
#     plt.tight_layout()
#     plt.show(block=False)

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

    # Use the helper function to plot the data
    # plot_data(df['trunc_local_time'], {'px': df['px'], 'py': df['py'], 'pz': df['pz']}, 'Position Subplots', 'Time (s)', 'Position (m)')
    # plot_data(df['trunc_local_time'], {'vx': df['vx'], 'vy': df['vy'], 'vz': df['vz']}, 'Velocity Subplots', 'Time (s)', 'Velocity (m/s)')
    # plot_data(df['trunc_local_time'], {'roll': df['roll'], 'pitch': df['pitch'], 'yaw': df['yaw']}, 'Orientation Subplots', 'Time (s)', 'Orientation (rad)')
    # plot_data(df['trunc_local_time'], {'rollRate': df['rollRate'], 'pitchRate': df['pitchRate'], 'yawRate': df['yawRate']}, 'Angular Velocity Subplots', 'Time (s)', 'Angular Velocity (rad/s)')

    # plot_fft_subplots(df['trunc_local_time'].values, {'px': df['px'].values, 'py': df['py'].values, 'pz': df['pz'].values}, 'Position FFT Subplots')
    # plot_fft_subplots(df['trunc_local_time'].values, {'vx': df['vx'].values, 'vy': df['vy'].values, 'vz': df['vz'].values}, 'Velocity FFT Subplots')
    # plot_fft_subplots(df['trunc_local_time'].values, {'roll': df['roll'].values, 'pitch': df['pitch'].values, 'yaw': df['yaw'].values}, 'Orientation FFT Subplots')
    # plot_fft_subplots(df['trunc_local_time'].values, {'rollRate': df['rollRate'].values, 'pitchRate': df['pitchRate'].values, 'yawRate': df['yawRate'].values}, 'Angular Velocity FFT Subplots')

    # Use the helper function to plot the PSDs
    # plot_psd_subplots(df['trunc_local_time'].values, {'px': df['px'].values, 'py': df['py'].values, 'pz': df['pz'].values}, 'Position PSD Subplots', fs)
    # plot_psd_subplots(df['trunc_local_time'].values, {'vx': df['vx'].values, 'vy': df['vy'].values, 'vz': df['vz'].values}, 'Velocity PSD Subplots', fs)
    # plot_psd_subplots(df['trunc_local_time'].values, {'roll': df['roll'].values, 'pitch': df['pitch'].values, 'yaw': df['yaw'].values}, 'Orientation PSD Subplots', fs)
    # plot_psd_subplots(df['trunc_local_time'].values, {'rollRate': df['rollRate'].values, 'pitchRate': df['pitchRate'].values, 'yawRate': df['yawRate'].values}, 'Angular Velocity PSD Subplots', fs)
    

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
    plot_box_psd(collection_dfs, [inspection_var], collection_fs, collection_filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Frequency Metric for Centralized and Decentralized Data.')
    parser.add_argument('--centralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--decentralized_file_path', type=str, help='path to the file.')
    parser.add_argument('--mode', type=str, help='')
    args = parser.parse_args()
    main(args)
    input()
