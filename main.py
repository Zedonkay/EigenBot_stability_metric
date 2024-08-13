import pandas as pd
import lyapunov_final as lyap
import numpy as np
import truncate as tr
import state_space as ss
import psd as psd

#!/path/to/venv python3

print("Running main.py")

def main():
    # Set the values for various parameters
    tau = 1  # Time delay for Lyapunov exponent calculation
    m = 3  # Embedding dimension for Lyapunov exponent calculation
    delta_t = 0.01  # Time step for Lyapunov exponent calculation
    min_steps = 100  # Minimum number of steps for Lyapunov exponent calculation
    force_minsteps = False  # Flag to force minimum steps for Lyapunov exponent calculation
    epsilon = 10  # Tolerance for truncation and re-truncation
    plotting_0 = 0  # Starting point for Lyapunov exponent plotting
    plotting_final = 300  # Ending point for Lyapunov exponent plotting
    tolerance = 0.05*np.pi  # Tolerance for truncation and re-truncation

    # Initialize lists to store results
    neural_exponents = []  # List to store neural control Lyapunov exponents
    predefined_exponents = []  # List to store predefined control Lyapunov exponents
    neural_terrains= []  # List to store neural control terrains
    predefined_terrains = []  # List to store predefined control terrains


    # Keep track of PSDs (Power Spectral Densities)
    psds_neural = []  # List to store neural control PSDs
    psds_predefined = []  # List to store predefined control PSDs

    # Read data from a CSV file
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()

    # Process each file in the data
    for file in data:
        print(f"Processing data for {file[0]} control on {file[1]} terrain (leg {file[2]})")
        
        # Store frequencies
        if file[1] == "Neural":
            neural_terrains.append(file[0])
        else:
            predefined_terrains.append(file[0])
        #truncate the data
        tr.retruncate(file[0], file[1], file[2], file[3],file[4])

        ss.plot_time_differences(file[0], file[1], file[2])

        # Calculate Lyapunov exponents for the data

        lyap.exponent(tau, m, min_steps, epsilon, plotting_0, plotting_final,
                      delta_t, force_minsteps, neural_exponents,predefined_exponents ,file[0], file[1], file[2])
        
        # # Calculate PSDs for the data
        # psd.main(psds_neural,psds_predefined, file[0], file[1], file[2])

    # Plot the Lyapunov exponents
    print("plotting lyapunov exponents")
    lyap.plot_exponents(predefined_exponents, neural_exponents, neural_terrains)

    # # Plot the PSDs
    # print("plotting psdss")
    # psd.plot_psd(psds_neural, psds_predefined, neural_terrains)

    # # Save the results to CSV files
    # print("saving data")
    # data = pd.DataFrame(np.column_stack((predefined_terrains, predefined_exponents)),
    #                     columns=['frequency', 'exponent'])
    # data.to_csv("3_results/Flat/Flat_exponents.csv", index=False)

    # data = pd.DataFrame(np.column_stack((hill_trial, hill_exponents)))
    # data.to_csv("3_results/Hill/Hill_exponents.csv", index=False)
    


if __name__ == "__main__":
    main()
