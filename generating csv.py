import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def lorentz_system(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def main():
    # Parameters for the Lorentz system
    sigma = 16
    rho = 45.92
    beta = 8.0 / 2.0
    
    # Initial state of the system [x, y, z]
    initial_state = [1,1,1]
    
    # Number of steps for data set
    num_steps = 10000
    
    # Time step for creation of data set
    delta_t = 0.01
    
    positions = np.empty((num_steps+1,3))
    positions[0]= initial_state

    for i in range(num_steps):
        positions[i+1] = positions[i] + delta_t*lorentz_system(positions[i],0,sigma,rho,beta)
    
    data = pd.DataFrame(positions,columns=['px','py','pz'])
    data.to_csv('lorentz_data.csv',index=False)
  
if __name__ == '__main__':
    main()