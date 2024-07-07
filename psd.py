import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg
from scipy.signal import welch
import matplotlib.patches as mpatches
import matplotlib.markers as markers

labels=[]
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

def plot_psd(centralised_frequencies,centralised_psds,distributed_frequencies,distributed_psds):
    fig,ax=plt.subplots()
    add_label(ax.violinplot(centralised_psds,side='high',showmeans=False,showmedians=False,showextrema=False),"Centralised")
    add_label(ax.violinplot(distributed_psds,side='low',showmeans=False,showmedians=False,showextrema=False),"Distributed")
    ax.set_xticks(distributed_frequencies)
    for i in range(len(distributed_frequencies)):
        if(i<len(centralised_frequencies)):
            ax.scatter(centralised_frequencies[i],np.mean(centralised_psds[i]),marker = markers.CARETLEFTBASE, color = 'r')
            ax.scatter(centralised_frequencies[i],np.min(centralised_psds[i]),marker = markers.TICKLEFT, color = 'r')
            ax.scatter(centralised_frequencies[i],np.max(centralised_psds[i]),marker = markers.TICKLEFT, color = 'r')
        ax.scatter(distributed_frequencies[i],np.mean(distributed_psds[i]),marker = markers.CARETRIGHTBASE, color='b')
        ax.scatter(distributed_frequencies[i],np.min(distributed_psds[i]),marker = markers.TICKRIGHT, color='b')
        ax.scatter(distributed_frequencies[i],np.max(distributed_psds[i]),marker = markers.TICKRIGHT, color='b')
    ax.legend(*zip(*labels))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD*Freq")
    fig.savefig("6_Results/clean_data/psd.png")
    plt.clf()
    plt.close()

def calc(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    return f, Pxx

def main(psds_centralised,psds_distributed,frequency,test,control_type):
    filename = fg.filename_clean(frequency,test,control_type)
    df = pd.read_csv(filename)
    pdata = df[['az']]
    data=pdata.values
    f, Pxx = calc(data)
    if control_type == "centralised":
        psds_centralised.append(Pxx*f)
    else:
        psds_distributed.append(Pxx*f)