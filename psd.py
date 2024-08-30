import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg
from scipy.signal import welch
import matplotlib.patches as mpatches



def plot_psd(psds,date,terrain,trial):
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    
    ax.set_xticklabels(psds.keys())
    
    ax.violinplot(psds.values(), showmeans=False, showmedians=False, showextrema=False)

    ax.set_xlabel("trial")
    ax.set_ylabel("PSD*Freq")
    ax.set_ylabel("PSD*Freq")

    ax.set_title("PSD*Freq for Z-Acceleration on Flat Terrain")
    fig.savefig(fg.filename_big(date,terrain,trial)+"psd.png")
    plt.clf()
    plt.close()

def calc(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    return f, Pxx

def main(psds,date,terrain,trial):
    filename = fg.filename_clean_data(date,terrain,trial)
    df = pd.read_csv(filename)
    pdata = df[['az']]
    data=pdata.values
    f, Pxx = calc(data)
    psds.update({trial: f*Pxx})
    