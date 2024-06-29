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

# Main function
def main():
    # List of parameters for truncation
    recheck = [
        [370,"2", "distributed", 400, 9999],
        [400, "2", "distributed", 0, 980],
        [700, "1", "distributed", 900, 9999],
        [1040, "1", "distributed", 0, 1500],
        [1120, "1", "distributed", 0, 1500],
        [1160, "1", "distributed", 0, 1550],
        [1200, "1", "distributed", 650, 9999], 
        [1440, "1", "distributed", 200, 9999],
        [1600,"1", "distributed", 200, 9999],
        [1800, "1", "distributed", 200, 9999],
        [100, "1", "centralised", 0, 1380],
        [140, "1", "centralised", 0, 1380],
        [180, "1", "centralised", 100, 9999],
        [180, "2", "centralised", 0, 1380],
        [220, "2", "centralised", 0, 1150],
        [220, "3", "centralised", 180, 9999],
        [260, "3", "centralised", 0, 1530],
        [320, "1", "centralised", 0, 1450],
        [320, "2", "centralised", 20, 780]
    ]
    
    # Iterate over the list and truncate the data for each set of parameters
    for elem in recheck:
        truncate(elem[0], elem[1], elem[2], elem[3], elem[4])

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()