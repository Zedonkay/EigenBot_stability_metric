import pandas as pd
import lyapunov_final as lyap
import filename_generation as fg
import numpy as np
import truncate as tr
import retruncate as rtr
import state_space as ss
import matplotlib.pyplot as plt

def main():
    # Set the values for various parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    g_0 = 0
    g_f = 300
    epsilon = 10
    tolerance = 0.001

    # Initialize lists to store results
    centralised_exponents = []
    centralised_frequencies = []
    distributed_exponents = []
    distributed_frequencies = []

    # Read data from a CSV file
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()

    # Process each file in the data
    for file in data:
        print(f"{file[0]} {file[1]}hz test {file[2]}")

        # Perform truncation on the data
        tr.main(file[1], file[2], file[0], tolerance)

        # Perform re-truncation on the data
        rtr.truncate(file[1], file[2], file[0], file[3], file[4])

        # Perform state space analysis on the data
        ss.main(file[1], file[0], file[2])

        # Calculate Lyapunov exponents for the data
        lyap.exponent(tau, m, min_steps, epsilon, g_0, g_f, file[5], file[6],
                      delta_t, force_minsteps, centralised_frequencies,
                      centralised_exponents, distributed_frequencies,
                      distributed_exponents, file[1], file[0], file[2])

    # Plot the Lyapunov exponents
    lyap.plot_exponents(centralised_frequencies, centralised_exponents,
                        distributed_frequencies, distributed_exponents)

    # Save the results to CSV files
    data = pd.DataFrame(np.column_stack((centralised_frequencies, centralised_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/centralised/centralised_exponents.csv", index=True)

    data = pd.DataFrame(np.column_stack((distributed_frequencies, distributed_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/distributed/distributed_exponents.csv", index=True)

if __name__ == "__main__":
    main()