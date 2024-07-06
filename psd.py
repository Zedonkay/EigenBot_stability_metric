import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filename_generation as fg
from scipy.signal import welch

def plot_psd(centralised_frequencies,centralised_psds,distributed_frequencies,distributed_psds):
    fix,axs = plt.subplots(1,2,figsize=(10,6))
    axs[0].boxplot(centralised_psds)
    axs[0].set_xticklabels(centralised_frequencies)
    axs[0].set_xlabel("Update Rate [Hz]")
    axs[0].set_ylabel("Power Spectrum Density * Freq [W*Hz]")
    axs[0].set_ylim([0,2])
    axs[0].set_title("Centralised")
    axs[1].boxplot(distributed_psds)
    axs[1].set_xticklabels(distributed_frequencies)
    axs[1].set_xlabel("Update Rate [Hz]")
    axs[1].set_ylabel("Power Spectrum Density * Freq [W*Hz]")
    axs[1].set_ylim([0,2])
    axs[1].set_title("Distributed")
    title = "PSD*Freq Distribution for Z-Position vs. Update Rate"
    plt.savefig("6_Results/clean_data/psd.png")
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