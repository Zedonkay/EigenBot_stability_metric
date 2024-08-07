# EigenBot Stability Metrics

Documentation on Stability Metrics for the EigenBot
Motivation: The EigenBot is a dynamical system with many hyperparameters. In formulating our own metric, the team is able to hopefully create comparisons between EigenBot and existing systems, and more importantly, understand the effects of hyperparameters on the dynamical system.

Various tactics include understanding the "shakiness" (via FFT/Power Sensitivity Distribution), "sensitivity" (via Gait Sensitivity Norm), and "stability" (via Lyapunov Exponents) of EigenBot's gaits.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Following are dependencies for `main.py`:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
```
## Usage
set parameters for running lyapunov in main.py  
adjust file by file parameters in 2_raw_data/running_info.csv  
run the main.py file  
to run against all data that looks good (multiple files per frequency) change line 36 in main.py to df = pd.read_csv("2_raw_data/running_info_for_all.csv")  
to create correct directories to run for all data, delete all directories inside of 1_clean_data and 6_results/clean_data and run create_directories.py with its line 7 changed to df = pd.read_csv("2_raw_data/running_info_for_all.csv")


## Contact
Repository created 4/22/24
hkou@andrew.cmu.edu

Fork created 6/19/24
ishikhar@andrew.cmu.edu

