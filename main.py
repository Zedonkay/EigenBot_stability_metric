import pandas as pd
import lyapunov_final as lyap
import numpy as np
import process_data as tr
import state_space as ss
import psd as psd
import filename_generation as fg
import fft as fft
import process_data as pr_d
import create_directories as cd

#!/path/to/venv python3

print("Running main.py")

def main():
    # Set the values for various parameters
    tau = 1  # Time delay for Lyapunov exponent calculation
    m = 3  # Embedding dimension for Lyapunov exponent calculation
    delta_t = 1  # Time step for Lyapunov exponent calculation
    min_steps = 100  # Minimum number of steps for Lyapunov exponent calculation
    force_minsteps = False  # Flag to force minimum steps for Lyapunov exponent calculation
    epsilon = 10  # Tolerance for truncation and re-truncation
    plotting_0 = 0  # Starting point for Lyapunov exponent plotting
    plotting_final = 300  # Ending point for Lyapunov exponent plotting
    tolerance = 0.05*np.pi  # Tolerance for truncation and re-truncation

    # Initialize lists to store results
    lyapunov_exponents = []  # List to store Lyapunov exponents
    legs = []  # List to store leg numbers


    # Keep track of PSDs (Power Spectral Densities)
    psds=[]

    # # Read data from a CSV file
    # df = pd.read_csv("1_clean_data/running_info.csv")
    # data = df.to_numpy()

    data = list(range(1, 7))

    #Process the data
    for file in data:
        print("Processing Leg: ", file)
        # Read data from a CSV file
        df = pd.read_csv(fg.filename_clean(file))
        data = df.to_numpy()

        # Calculate the time differences between each timestamp
        time_diff = np.diff(data[:, 0])

        # Calculate the mean time difference
        delta_t = np.mean(time_diff)
        print("Mean time difference: ", delta_t)

        #Plotting Data
        print("Plotting Data")
        ss.plot_data(file)

        # Calculate the Lyapunov exponents
        print("Calculating Lyapunov Exponents")
        lyap.exponent(tau, m, min_steps, epsilon, plotting_0, plotting_final, delta_t, force_minsteps, lyapunov_exponents, file)

        # Calculate the PSD
        print("Calculating PSD")
        psd.main(psds, file)

        # Calculate the FFT
        print("Calculating FFT")
        fft.main(file)

        # Append the leg number to the list
        legs.append(file)

    # Plot the Lyapunov exponents
    print("Plotting Lyapunov Exponents")
    lyap.plot_exponents(lyapunov_exponents, legs)

    #plot the PSDs
    print("Plotting PSDs")
    psd.plot_psd(psds, legs)
        


if __name__ == "__main__":
    main()
