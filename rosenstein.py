import numpy as np

def reconstruction(data, tau, m):
    """
    Reconstruct the data using time delay embedding.

    Args:
        data (ndarray): The input data.
        tau (int): The time delay.
        m (int): The embedding dimension.

    Returns:
        ndarray: The reconstructed data.
    """
    d = len(data) - (m - 1) * tau
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
    """
    Find the closest vectors for the Rosenstein method.

    Args:
        reconstructed_data (ndarray): The reconstructed data.
        min_step (int): The minimum step size.
        t_f (int): The final time.

    Returns:
        list: The indices of the closest vectors.
    """
    neighbors = []
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
            neighbors_index.append(-500)
    
    return neighbors_index


def log_distance(reconstructed_data, neighbors_index, i):
    """
    Calculate the expected log distance for the Rosenstein method.

    Args:
        reconstructed_data (ndarray): The reconstructed data.
        neighbors_index (list): The indices of the closest vectors.
        i (int): The current time.

    Returns:
        float: The expected log distance.
    """
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
    """
    Calculate the Lyapunov exponents using the Rosenstein method.

    Args:
        data (ndarray): The input data.
        tau (int): The time delay.
        m (int): The embedding dimension.
        min_steps (int): The minimum number of steps.
        t_0 (int): The initial time.
        t_f (int): The final time.
        delta_t (float): The time step size.

    Returns:
        ndarray: The times.
        ndarray: The mean log distances.
    """
    reconstructed_data = reconstruction(data, tau, m)
    neighbors_index = find_closest_vectors(reconstructed_data, min_steps, t_f)
    mean_log_distance = []
    times = []
    
    for i in range(t_0, t_f):
        mean_log_distance.append(log_distance(reconstructed_data, neighbors_index, i))
        times.append(i * delta_t)
    
    mean_log_distance = np.array(mean_log_distance)
    times = np.array(times)
    
    return times, mean_log_distance
