# EigenBot Stability Metrics

Documentation on Stability Metrics for the EigenBot
Motivation: The EigenBot is a dynamical system with many hyperparameters. In formulating our own metric, the team is able to hopefully create comparisons between EigenBot and existing systems, and more importantly, understand the effects of hyperparameters on the dynamical system.

Various tactics include understanding the "shakiness" (via FFT/Power Sensitivity Distribution), "sensitivity" (via Gait Sensitivity Norm), and "stability" (via Lyapunov Exponents) of EigenBot's gaits.

## Table of Contents

- [Installation](#installation)

## Installation

Following are dependencies for `lyapunov_final.py`:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import filename_generation as fg
```

## Contact
Repository created 4/22/24
hkou@andrew.cmu.edu

Fork created 6/19/24
ishikhar@andrew.cmu.edu

