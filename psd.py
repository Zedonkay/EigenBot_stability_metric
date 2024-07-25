import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg
from scipy.signal import welch
import matplotlib.patches as mpatches



def plot_psd(types,psds):
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    
    ax.set_xticklabels(types)
    
    ax.violinplot(psds,showmeans=False,showmedians=False,showextrema=False)

    ax.set_xlabel("Type")
    ax.set_ylabel("PSD*Freq")
    ax.set_ylabel("PSD*Freq")

    ax.set_title("PSD*Freq for Z-Acceleration on Flat Terrain")
    fig.savefig("3_Results/psd.png")
    plt.clf()
    plt.close()

def calc(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    return f, Pxx

def main(psds,type):
    filename = fg.filename_clean_data(type)
    df = pd.read_csv(filename)
    pdata = df[['az']]
    data=pdata.values
    f, Pxx = calc(data)
    psds.append(f*Pxx)
    