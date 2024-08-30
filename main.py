import pandas as pd  # Import the pandas library for data manipulation
import lyapunov_final as lyap  # Import the lyapunov_final module
import numpy as np  # Import the numpy library for numerical operations
import truncate as tr  # Import the truncate module
import state_space as ss  # Import the state_space module
import psd as psd  # Import the psd module
import filename_generation as fg  # Import the filename_generation module

#!/path/to/venv python3

print("Running main.py")

def main():
    # Set the values for various parameters
    tau = 11  # Time delay for Lyapunov exponent calculation (replaced in function)
    m = 11  # Embedding dimension for Lyapunov exponent calculation
    delta_t = 0.01  # Time step for Lyapunov exponent calculation
    min_steps = 100  # Minimum number of steps for Lyapunov exponent calculation
    force_minsteps = False  # Flag to force minimum steps for Lyapunov exponent calculation
    epsilon = 10  # Tolerance for truncation
    plotting_0 = 0  # Starting point for plotting
    plotting_final = 300  # Ending point for plotting
    tolerance = 0.001  # Tolerance for truncation

    df = pd.read_csv("1_raw_data/running_info.csv")  # Read the CSV file into a pandas DataFrame
    data = df.values  # Convert the DataFrame to a numpy array
    dates = df['Date'].values  # Get the dates from the 'Date' column
    dates = np.unique(dates)  # Get the unique dates
    terrains = {}  # Initialize an empty dictionary to store terrains
    trials = {}  # Initialize an empty dictionary to store trials

    for date in dates:
        terrains.update({date: np.unique(data[data[:, 0] == date][:, 1])})  # Get the unique terrains for each date
   
    for date in dates:
        trials.update({date: {}})
        for terrain in terrains.get(date):
            trials.get(date).update({terrain: np.unique(data[(data[:, 0] == date) & (data[:, 1] == terrain)][:, 2])})  # Get the unique trials for each date and terrain
    for date in dates:
        for terrain in terrains.get(date):
            exponents = {}  # Initialize an empty list to store Lyapunov exponents
            psds = {}  # Initialize an empty list to store power spectral densities
            forward_velocities = {}  # Initialize an empty list to store forward velocities
            forward_velocities2 = {}  # Initialize an empty list to store forward velocities (for the second method)

            for trial in trials.get(date).get(terrain):
                print("Running for date:", date, ", terrain:", terrain, ", trial:", trial)
                row = df[(df['Date'] == date) & (df['Terrain'] == terrain) & (df['Trial'] == trial)]  # Get the row corresponding to the current date, terrain, and trial
                row_array = row.values[0]  # Convert the row to a numpy array
                tr_start = row_array[3]  # Get the start time for truncation
                tr_end = row_array[4]  # Get the end time for truncation

                tr.main(date, terrain, trial, tolerance)  # Call the main function from the truncate module
                tr.retruncate(date, terrain, trial, tr_start, tr_end)  # Call the retruncate function from the truncate module

                info = pd.read_csv(fg.filename_clean_data(date, terrain, trial))  # Read the cleaned data from the CSV file
                delta_t = np.mean(np.diff(info['timestamp'])) # Calculate the mean time step

                ss.main(date, terrain, trial, delta_t,forward_velocities,forward_velocities2)  # Call the main function from the state_space module
                lyap.exponent(tau, m, min_steps, epsilon, plotting_0, plotting_final,
                              delta_t, force_minsteps, exponents, date, terrain, trial)  # Call the exponent function from the lyapunov_final module
                psd.main(psds, date, terrain, trial)  # Call the main function from the psd module

            lyap.plot_exponents(exponents, date, terrain, 0)  # Call the plot_exponents function from the lyapunov_final module
            psd.plot_psd(psds, date, terrain, 0)  # Call the plot_psd function from the psd module

            forward_velocities_df = pd.DataFrame.from_dict(forward_velocities, orient='index')
            forward_velocities_df.to_csv(fg.filename_big(date, terrain, 0) + "forward_velocities.csv")  # Save the DataFrame to a CSV file

            forward_velocities_df2 = pd.DataFrame.from_dict(forward_velocities2, orient='index')
            forward_velocities_df2.to_csv(fg.filename_big(date, terrain, 0) + "forward_velocities2.csv")  # Save the DataFrame to a CSV file

            exponents_df = pd.DataFrame.from_dict(exponents, orient='index')
            exponents_df.to_csv(fg.filename_big(date, terrain, 0) + "exponents.csv")  # Save the DataFrame to a CSV file

            psds_df = pd.DataFrame.from_dict(psds, orient='index')
            psds_df.to_csv(fg.filename_big(date, terrain, 0) + "psds.csv")  # Save the DataFrame to a CSV file


if __name__ == "__main__":
    main()  # Call the main function if the script is run directly
