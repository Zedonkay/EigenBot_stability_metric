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
    tau = 3  # Time delay for Lyapunov exponent calculation
    m = 11  # Embedding dimension for Lyapunov exponent calculation
    delta_t = 0.01  # Time step for Lyapunov exponent calculation
    min_steps = 100  # Minimum number of steps for Lyapunov exponent calculation
    force_minsteps = False  # Flag to force minimum steps for Lyapunov exponent calculation
    epsilon = 10  # Tolerance for truncation and re-truncation
    plotting_0 = 0  # Starting point for Lyapunov exponent plotting
    plotting_final = 300  # Ending point for Lyapunov exponent plotting
    tolerance = 0.001  # Tolerance for truncation and re-truncation

    # Initialize lists to store results
    predefined_exponents = []  # List to store predefined Lyapunov exponents
    predefined_disturbances = []  # List to store predefined disturbances
    neural_exponents = []  # List to store neural Lyapunov exponents
    neural_disturbances = []  # List to store neural disturbances

    # Keep track of PSDs (Power Spectral Densities)
    psds_neural = []  # List to store neural PSDs
    psds_predefined = []  # List to store predefined PSDs

    # Read data from a CSV file
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()

    # Process each file in the data
    for file in data:
        print(f"Processing {file[0]} control for {file[1]}")
        
        # Store frequencies
        if file[0] == "Predefined":
            predefined_disturbances.append(file[1])
        else:
            neural_disturbances.append(file[1])

        # Perform truncation on the data
        tr.main(file[1], file[0], tolerance)

        # Perform re-truncation on the data
        tr.retruncate(file[1], file[0], file[2], file[3])

        # Perform state space analysis on the data
        ss.main(file[1], file[0])

        # Calculate Lyapunov exponents for the data
        lyap.exponent(tau, m, min_steps, epsilon, plotting_0, plotting_final,
                      delta_t, force_minsteps, predefined_exponents,
                      neural_exponents, file[1], file[0])

        # Calculate PSDs for the data
        psd.main(psds_neural, psds_predefined, file[1], file[0])

    # Plot the Lyapunov exponents
    lyap.plot_exponents(predefined_exponents, neural_exponents, predefined_disturbances)

    # Plot the PSDs
    psd.plot_psd(psds_neural, psds_predefined, neural_disturbances)

    # Save the results to CSV files
    data = pd.DataFrame(np.column_stack((predefined_disturbances, predefined_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("3_results/predefined_exponents.csv", index=True)

    data = pd.DataFrame(np.column_stack((neural_disturbances, neural_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("3_results/neural_exponents.csv", index=True)


if __name__ == "__main__":
    main()
