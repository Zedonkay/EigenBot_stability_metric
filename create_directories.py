import os
import pandas as pd
import numpy as np
import shutil

def main():
    if(os.path.exists("2_results")):
            shutil.rmtree("2_results")
    os.makedirs("2_results")
    if(os.path.exists("1_clean_data")):
        shutil.rmtree("1_clean_data")
    os.makedirs("1_clean_data")
    for i in range(1,7):
        os.makedirs("2_results/leg"+str(i))

if __name__ == "__main__":
    main()
