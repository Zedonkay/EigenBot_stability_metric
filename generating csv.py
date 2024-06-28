import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def lorentz_system(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])
def main():
    # Parameters for the Lorentz system
    sigma = 10
    rho = 28
    beta = 8.0 / 3.0
    
   
    
    # Number of steps for data set
    num_steps = 10000

    # Time step for creation of data set
    delta_t = 0.01

    #initial state
    positions = np.zeros((num_steps+1,3))
    positions[0] = (0, 1, 1.05)
    times = np.arange(0, (num_steps+1)*delta_t, delta_t)

    for i in range(num_steps):
        positions[i+1]=positions[i]+lorentz_system(positions[i],sigma,rho,beta)*delta_t
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*positions.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()
    data = pd.DataFrame(positions,index=times,columns=['px','py','pz'])
    data.to_csv('lorentz_data.csv',index=False)
  
if __name__ == '__main__':
    main()