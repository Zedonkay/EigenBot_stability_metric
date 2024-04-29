# EigenBot Stability Metrics

Documentation on Stability Metrics for the EigenBot
Motivation: The EigenBot is a dynamical system with many hyperparameters. In formulating our own metric, the team is able to hopefully create comparisons between EigenBot and existing systems, and more importantly, understand the effects of hyperparameters on the dynamical system.

Various tactics include understanding the "shakiness" (via FFT/Power Sensitivity Distribution), "sensitivity" (via Gait Sensitivity Norm), and "stability" (via Lyapunov Exponents) of EigenBot's gaits.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Following are dependencies for `frequency_metric.py`:
```
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.signal import find_peaks
import os
import natsort
import ipdb
```

## Usage

To run `frequency_metric.py`

`python3 frequency_metric.py --centralized_file_path "sample_data/centralised/*.csv" --decentralized_file_path "sample_data/distributed/*.csv"`

For the clean dataset

`python3 frequency_metric.py --centralized_file_path "clean_data/centralised/*.csv" --decentralized_file_path "clean_data/distributed/*.csv"`

For neural v predefined dataset
`python3 frequency_metric.py --centralized_file_path "raw_predefined_v_neural_data/predefined/*.csv" --decentralized_file_path "raw_predefined_v_neural_data/neural/*.csv"`

## Contact
Repository created 4/22/24
hkou@andrew.cmu.edu

