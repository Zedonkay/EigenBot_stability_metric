import numpy as np

def autocorr4(x):
    '''
    Calculate autocorrelation using Fast Fourier Transform (FFT).

    Parameters:
    - x: numpy array, input data
    - lags: list, list of lag values for which to calculate autocorrelation

    Returns:
    - corr: numpy array, autocorrelation values corresponding to the given lags
    '''
    mean = x.mean()
    var = np.var(x)
    xp = x - mean

    cf = np.fft.fft(xp)
    sf = cf.conjugate() * cf
    corr = np.fft.ifft(sf).real / var / len(x)
    return corr


def find_lag(data, min_steps):
    '''
    Find the lag value that maximizes the autocorrelation of the given data.

    Parameters:
    - data: numpy array, input data
    - min_steps: int, minimum number of steps to consider for lag search

    Returns:
    - lag: int, optimal lag value that maximizes the autocorrelation
    '''
    correlation = autocorr4(data)
    lag = np.argmax(correlation)+1
    correlation = correlation[int(min_steps*.08):int(min_steps*.55)]
    used_lag= np.argmax(correlation)+1+int(min_steps*.08)
    print("True lag: ",lag, "Used lag: ",used_lag)
    return used_lag


def reconstruction(data, tau, m):
    '''
    Reconstruct the data using time delay embedding.

    Parameters:
    - data: numpy array, input data
    - tau: int, time delay between consecutive samples
    - m: int, embedding dimension

    Returns:
    - reconstructed_data: numpy array, reconstructed data using time delay embedding
    '''
    d = len(data)
    d = d - (m - 1) * tau
    if len(data.shape) == 1:
        reconstructed_data = np.empty((d, m))
    else:
        reconstructed_data = np.empty((d, m * len(data[0])))
    for i in range(d):
        for j in range(m):
            if len(data.shape) == 1:
                reconstructed_data[i][j] = data[i + j * tau]
            else:
                for k in range(len(data[0])):
                    reconstructed_data[i][j * len(data[0]) + k] = data[i + j * tau][k]
    return reconstructed_data


def find_closest_vectors(reconstructed_data, min_step, t_f):
    '''
    Find the closest vectors for Rosenstein method.

    Parameters:
    - reconstructed_data: numpy array, reconstructed data using time delay embedding
    - min_step: int, minimum step size between vectors
    - t_f: int, maximum index of vectors to consider

    Returns:
    - neighbors_index: list, indices of the closest vectors for each vector in the reconstructed data
    '''
    neighbors = []
    avg_dist = []
    neighbors_index = []
    for i in range(len(reconstructed_data)):
        closest_dist = -1
        ind = -1
        for j in range(len(reconstructed_data) - t_f):
            if i != j and abs(j - i) > min_step:
                dist = np.linalg.norm(reconstructed_data[i] - reconstructed_data[j])
                if closest_dist == -1 or dist < closest_dist:
                    ind = j
                    closest_dist = dist

        if closest_dist > 0 and closest_dist < 1e308 and not np.isnan(closest_dist):
            neighbors.append(np.log(closest_dist))
            neighbors_index.append(ind)
        elif closest_dist == 0:
            neighbors.append(0)
            neighbors_index.append(-500)
        else:
            print(closest_dist)
            neighbors_index.append(-500)
    return neighbors_index


def log_distance(reconstructed_data, neighbors_index, i) -> float:
    '''
    Calculate the expected log distance for Rosenstein method.

    Parameters:
    - reconstructed_data: numpy array, reconstructed data using time delay embedding
    - neighbors_index: list, indices of the closest vectors for each vector in the reconstructed data
    - i: int, index of the vector for which to calculate the log distance

    Returns:
    - log_dist: float, expected log distance for the given vector index
    '''
    d_ji = []
    for j in range(len(reconstructed_data) - i):
        if neighbors_index[j] == -500:
            print("error")
        else:
            if j + i < len(reconstructed_data) and neighbors_index[j] + i < len(reconstructed_data):
                d_ji.append(np.linalg.norm(reconstructed_data[neighbors_index[j] + i] - reconstructed_data[j + i]))
            else:
                print(j, i)
    d_ji = np.array(d_ji)
    return np.mean(np.log(d_ji))


def lyapunov(data, tau, m, min_steps, t_0, t_f, delta_t):
    '''
    Calculate the Lyapunov exponents using the Rosenstein method.

    Parameters:
    - data: numpy array, input data
    - tau: int, time delay between consecutive samples
    - m: int, embedding dimension
    - min_steps: int, minimum number of steps to consider for lag search
    - t_0: int, starting index for log distance calculation
    - t_f: int, ending index for log distance calculation
    - delta_t: float, time interval between consecutive indices

    Returns:
    - times: numpy array, time values corresponding to the calculated mean log distances
    - mean_log_distance: numpy array, mean log distances for each time value
    '''

    tau=find_lag(data, min_steps)
    
    # Reconstruction through time delay
    reconstructed_data = reconstruction(data, tau, m)
    # Find closest vectors
    neighbors_index = find_closest_vectors(reconstructed_data, min_steps, t_f)
    # Calculate mean distance
    mean_log_distance = []
    times = []

    

    for i in range(t_0, t_f):
        mean_log_distance.append(log_distance(reconstructed_data, neighbors_index, i))
        times.append(i * delta_t)
    mean_log_distance = np.array(mean_log_distance)
    times = np.array(times)
    # Calculate Lyapunov exponents
    return times, mean_log_distance