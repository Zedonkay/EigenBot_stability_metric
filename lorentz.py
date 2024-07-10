import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lyapunov_final as lf
import rosenstein

def plot_growth_factors(t_0,t_f):
    df = pd.read_csv('lorentz_lyapunovdata.csv')
    data = df['Mean Divergence'].values
    times=df['times'].values

    coef=np.polyfit(times[t_0:t_f],data[t_0:t_f],1)
    print(coef[0])
    poly1d_fn = np.poly1d(coef)

    ax = plt.plot(times,data,label="Average divergence", color="blue")
    plt.plot(times[t_0:t_f], poly1d_fn(times[t_0:t_f]),label=f"Least Squares Line", color="red")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Log mean divergence")
    plt.title("Log mean Divergence vs Time for Lorentz Attractor")
    plt.savefig("lorentz_lyapunov_plot.png")
    plt.clf()
    plt.close()

def lorentz_system(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])
def create_lorentz_data(num_steps, delta_t, sigma, rho, beta):
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
    plt.savefig("lorentz_attractor.png")
    plt.clf()
    plt.close()

    data = pd.DataFrame(positions,index=times,columns=['px','py','pz'])
    data.to_csv('lorentz_data.csv',index=False)

def lyapunov_exponents(tau, m,delta_t):
    df = pd.read_csv('lorentz_data.csv')
    data = df['pz'].values

    min_steps = lf.welch_method(data)
    if min_steps%1 != 0:
        min_steps = int(min_steps)+1
    else:
        min_steps = int(min_steps)
    
    t_0 = 0
    t_f = 2*min_steps


    times, data = rosenstein.lyapunov(data,tau,m,min_steps,0,3*min_steps,delta_t)

    data = pd.DataFrame(np.column_stack((times,data)),columns=['times','Mean Divergence'])
    data.to_csv('lorentz_lyapunovdata.csv',index=True)
    print(t_0, t_f)
    return t_0, t_f

    

def main():
    # Parameters for the Lorentz system
    sigma = 10
    rho = 28
    beta = 8.0 / 3.0

    # Number of steps for data set
    num_steps = 10000
    # Time step for creation of data set
    delta_t = 0.01

    #parameters for lyapunov
    tau = 1
    m = 3
    
    #create_lorentz_data(num_steps, delta_t, sigma, rho, beta)
    plot_growth_factors(0,200)
   

if __name__ == "__main__":
    main()