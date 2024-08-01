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

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def plot_psd(psds_neural,psds_predefined,disturbances):
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    
    
    add_label(ax.violinplot(psds_neural,side='high',showmeans=False,showmedians=False,showextrema=False),"Neural")
    add_label(ax.violinplot(psds_predefined,side='low',showmeans=False,showmedians=False,showextrema=False),"Predefined")
    
    
    ax.legend(*zip(*labels),loc=9)
    
    ax.set_xlabel("Disturbance")
    ax.set_ylabel("PSD*Freq")
    set_axis_style(ax, disturbances)
    ax.set_title("PSD*Freq for Z-Acceleration vs Frequency")
    fig.savefig(f"3_results/psd.png")
    plt.clf()
    plt.close()

def calc(data):
    data=np.reshape(data,(1,-1))
    time_series= data[0]
    f, Pxx = welch(time_series)
    return f, Pxx

def main(psds_neural,psds_predefined,disturbance,control_type):
    filename = fg.filename_clean(disturbance ,control_type)
    df = pd.read_csv(filename)
    pdata = df[['az']]
    data=pdata.values
    f, Pxx = calc(data)
    if control_type=="Predefined":
        psds_predefined.append(Pxx*f)
    else:
        psds_neural.append(Pxx*f)