#!/path/to/venv python3
import pandas as pd
import lyapunov_final as lyap
import filename_generation as fg
import numpy as np
import truncate as tr
import retruncate as rtr
import state_space as ss
import matplotlib.pyplot as plt
from scipy.signal import welch
import psd as psd

print("Running main.py")
def main():
    # Set the values for various parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    epsilon = 10
    plotting_0=0
    plotting_final=300
    tolerance = 0.001
    min = 120
    max = 1350

    # Initialize lists to store results
    centralised_exponents = []
    centralised_frequencies = []
    distributed_exponents = []
    distributed_frequencies = []

    # keep track of psds
    psds_centralised = []
    psds_distributed = []

    # Read data from a CSV file
    df = pd.read_csv("2_raw_data/running_info_all_good.csv")
    data = df.to_numpy()

    # Process each file in the data
    for file in data:
        print(f"{file[0]} {file[1]}hz test {file[2]}")
        # Store the frequency of the data
        if(file[0]=="centralised"):
            centralised_frequencies.append(file[1])
        else:
            distributed_frequencies.append(file[1])

        # Perform truncation on the data
        tr.main(file[1], file[2], file[0], tolerance)

        # Perform re-truncation on the data
        rtr.truncate(file[1], file[2], file[0], file[3], file[4])

        # Perform state space analysis on the data
        ss.main(file[1], file[0], file[2],min,max)
        
        # Calculate Lyapunov exponents for the data
        lyap.exponent(tau, m, min_steps, epsilon, plotting_0,plotting_final,
                      delta_t, force_minsteps, centralised_exponents, 
                      distributed_exponents, file[1], file[0], file[2])
        
        # Calculate PSDs for the data
        psd.main(psds_centralised, psds_distributed, file[1], file[2], file[0])

    # Plot the Lyapunov exponents
    lyap.plot_exponents(centralised_frequencies, centralised_exponents,
                        distributed_frequencies, distributed_exponents)

    # Plot the PSDs
    psd.plot_psd(centralised_frequencies, psds_centralised,
                 distributed_frequencies, psds_distributed)
    # Save the results to CSV files
    data = pd.DataFrame(np.column_stack((centralised_frequencies, centralised_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/centralised/centralised_exponents.csv", index=True)

    data = pd.DataFrame(np.column_stack((distributed_frequencies, distributed_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/distributed/distributed_exponents.csv", index=True)



if __name__ == "__main__":
    main()