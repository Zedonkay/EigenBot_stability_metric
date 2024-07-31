import pandas as pd
import numpy as np
import filename_generation as fg

# Function to truncate the data based on start and end indices
def truncate(frequency, test, control_type, start, end):
    # Generate the filename based on the input parameters
    filename = fg.filename_clean(frequency, test, control_type)
    
    # Read the raw data from the CSV file
    raw_test = pd.read_csv(filename)
    
    # Truncate the data based on the start and end indices
    if end != 9999:
        raw_test = raw_test.iloc[start:end]
    else:
        raw_test = raw_test.iloc[start:]
    
    # Save the truncated data back to the CSV file
    raw_test.to_csv(filename, index=False)

