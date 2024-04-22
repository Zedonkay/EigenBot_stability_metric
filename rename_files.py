import os
import glob
import re

# get the directory where the python script is running
dir_path = os.getcwd()

# iterate over all csv files in the directory
for filename in glob.glob(os.path.join(dir_path, '*.csv')):
    # check if "odometry_data_centralized_test_" is in the filename
    if "odometry_data_centralised_test_" in filename:
        # create new filename by removing "odometry_data_centralized_test_"
        new_filename = re.sub("odometry_data_centralised_test_", "", filename)
        # rename the file
        os.rename(filename, new_filename)
