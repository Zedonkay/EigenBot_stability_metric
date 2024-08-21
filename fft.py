import pandas as pd
import pynufft
import filename_generation as fg
import numpy as np
import matplotlib.pyplot as plt
def main(leg):
    """
    Function to calculate the Power Spectral Density (PSD) for a given leg.
    Args:
        leg (int): The leg number.
    """
    # Generate the filename for the given leg
    filename = fg.filename_clean(leg)
    
    # Read the data from the CSV file
    df = pd.read_csv(filename)
    data = df.to_numpy()

    # Sample data: replace this with your actual data
    timestamps = data[:, 0]  
    force_values = data[:,1]

    # Normalize timestamps to the range [0, 1]
    normalized_timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())

    # Reshape normalized_timestamps to be 2-dimensional
    normalized_timestamps = normalized_timestamps.reshape(-1, 1)

    # Initialize NUFFT object
    NufftObj = pynufft.NUFFT()

    # Define the shape of the signal
    Nd = (len(force_values),)  # Signal shape
    Kd = (2 * len(force_values),)  # Oversampled grid shape
    Jd = (6,)  # Interpolation points

    # Plan the NUFFT
    NufftObj.plan(normalized_timestamps, Nd, Kd, Jd)

    # Perform the NUFFT
    X_k = NufftObj.forward(force_values)

    # Perform the inverse NUFFT (optional, to reconstruct the signal)
    X_t = NufftObj.adjoint(X_k)

    #save the nuffft as a csv file
    df = pd.DataFrame(X_k)
    df.to_csv(fg.store_clean_data(leg)+'nufft.csv', index=False)


    # Create a scatter plot of the original signal
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.scatter(timestamps, force_values, label='Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.title('Original Signal')
    plt.legend()

    # Create a scatter plot of the NUFFT result (magnitude)
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(X_k)), np.abs(X_k), label='NUFFT Result (Magnitude)')
    plt.xlabel('Frequency Index')
    plt.ylabel('Magnitude')
    plt.title('NUFFT Result')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fg.store_clean_data(leg)+'nufft.png')
    plt.clf()
    plt.close()




main(1)
