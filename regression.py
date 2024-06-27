import numpy as np
import pandas as pd

def polyfit_csv(file_path, t_0, t_f):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract the times and data from the DataFrame
    times = df.index()
    data = df['lyapunov_exponent']

    # Perform polynomial fitting
    coef = np.polyfit(times[80:], data[80:], 1)
    poly1d_fn = np.poly1d(coef)

    # Return the fitted polynomial function
    return poly1d_fn
