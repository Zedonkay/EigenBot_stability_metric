#!/path/to/venv python3
import pandas as pd
import lyapunov_final as lyap
import numpy as np
import truncate as tr
import state_space as ss
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


    # Initialize lists to store results
    predefined_exponents = []
    predefined_disturbances = []
    neural_exponents = []
    neural_disturbances = []

    # keep track of psds
    psds_predefined = []
    psds_neural = []

    # Read data from a CSV file
    df = pd.read_csv("2_raw_data/running_info.csv")
    data = df.to_numpy()

    # Process each file in the data
    for file in data:
        print(f"Processing {file[0]} control for {file[1]}")
        #store frequencies
        if(file[0]=="predefined"):
            predefined_disturbances.append(file[1])
        else:
             s.append(file[1])

        # Perform truncation on the data
        tr.main(file[1],file[0], tolerance)

        # Perform re-truncation on the data
        tr.retruncate(file[1],file[0],file[2],file[3])

        # Perform state space analysis on the data
        ss.main(file[1],file[0])
        
        # Calculate Lyapunov exponents for the data
        lyap.exponent(tau, m, min_steps, epsilon, plotting_0,plotting_final,
                      delta_t, force_minsteps, predefined_exponents, 
                      neural_exponents, file[1], file[0])
        
        # Calculate PSDs for the data
        psd.main(psds_predefined, psds_neural, file[1], file[0])

    # # Plot the Lyapunov exponents
    # lyap.plot_exponents(predefined_disturbances, predefined_exponents,
    #                     neural_disturbances, neural_exponents)

    # Plot the PSDs
    psd.plot_psd(predefined_disturbances, psds_predefined,
                 neural_disturbances, psds_neural)
    # Save the results to CSV files
    data = pd.DataFrame(np.column_stack((predefined_disturbances, predefined_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/predefined/predefined_exponents.csv", index=True)

    data = pd.DataFrame(np.column_stack((neural_disturbances, neural_exponents)),
                        columns=['frequency', 'exponent'])
    data.to_csv("6_Results/clean_data/neural/neural_exponents.csv", index=True)



if __name__ == "__main__":
    main()