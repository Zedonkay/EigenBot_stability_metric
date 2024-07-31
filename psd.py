import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg
from scipy.signal import welch
import matplotlib.patches as mpatches

labels=[]
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

def plot_psd(centralised_frequencies,centralised_psds,distributed_frequencies,distributed_psds):
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    
    ax.set_xticklabels(distributed_frequencies)
    
    add_label(ax.violinplot(centralised_psds,side='high',showmeans=False,showmedians=False,showextrema=False), "Centralised Control")
    ax.set_xticklabels(centralised_frequencies)
    add_label(ax.violinplot(distributed_psds,side='low',showmeans=False,showmedians=False,showextrema=False), "Distributed Control")
    ax.set_xticklabels(distributed_frequencies)
    ax.legend(*zip(*labels),loc=9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD*Freq")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD*Freq")
    ax.set_title("PSD*Freq for Z-Acceleration vs Frequency on Flat Terrain")
    fig.savefig("6_Results/clean_data/psd.png")
    plt.clf()
    plt.close()

def calc(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    return f, Pxx

def main(psds_centralised,psds_distributed,disturbance,control_type):
    filename = fg.filename_clean(disturbance,control_type)
    df = pd.read_csv(filename)
    pdata = df[['az']]
    data=pdata.values
    f, Pxx = calc(data)
    if control_type == "centralised":
        psds_centralised.append(Pxx*f)
    else:
        psds_distributed.append(Pxx*f)