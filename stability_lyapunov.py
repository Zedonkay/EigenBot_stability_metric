import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rosenstein_method(data, m, tau, num_iterations):
    """Calculate Lyapunov exponents using the Rosenstein method."""
    n = len(data)
    d = len(data[0])

    # Initialize the Lyapunov exponents
    lyap_exponents = np.zeros(d)

    for i in range(d):
        # Choose a random orthonormal vector
        v = np.random.rand(d)
        v /= np.linalg.norm(v)

        sum_ = 0.0

        for j in range(num_iterations):
            # Calculate the Euclidean distance between the two trajectories
            dist = np.linalg.norm(data[j] - data[j + tau])

            # Update the sum of logarithms of distances
            sum_ += np.log(dist)

            # Orthogonalize the vector
            v = np.dot(np.eye(d) - np.outer(v, v), v)

            # Normalize the vector
            v /= np.linalg.norm(v)
       
        # Calculate the average sum of logarithms of distances
        lyap_exponents[i] = sum_ / (n - tau * num_iterations)

    return lyap_exponents

def plot_lyapunov_exponents(lyap_exponents):
    """Plot Lyapunov exponents."""
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(lyap_exponents)), lyap_exponents, color='skyblue')
    plt.xlabel('Exponent Index')
    plt.ylabel('Lyapunov Exponent Value')
    plt.title('Lyapunov Exponents')
    plt.show()


def main():
    # Load time series data from CSV
    df = pd.read_csv("clean_data/centralised/220Hz.csv")
    pdata = df[['px', 'py', 'pz']]
    data=pdata.values


    # Define parameters
    m = 3  # Embedding dimension
    tau = 1  # Time delay
    num_iterations = 100  # Number of iterations
    
    # Calculate Lyapunov exponents using the Rosenstein method
    lyap_exponents = rosenstein_method(data, m, tau, num_iterations)

    # Print the calculated Lyapunov exponents
    print("Lyapunov Exponents:")
    
    for i, exponent in enumerate(lyap_exponents):
        print(f"Exponent {i+1}: {exponent}")

    # Plot Lyapunov exponents
    plot_lyapunov_exponents(lyap_exponents)

if __name__ == "__main__":
    main()
