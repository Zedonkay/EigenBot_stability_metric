import numpy as np 
import pandas as pd
from scipy.integrate import solve_ivp
def lorentz(t,system,sigma,rho,beta):
    x = system[0]
    y = system[1]
    z = system[2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])
def Jacobian(system, sigma, rho, beta):
    x, y, z = system
    Jacobian = np.array([[-sigma, sigma, 0],
                  [rho - z, -1, -x],
                  [y, x, -beta]])
    return Jacobian
def appJac(system, derivs,t):
    dx, dy, dz = derivs
    J = Jacobian(system,t)
    dfx = np.dot(J,dx)
    dfy = np.dot(J,dy)
    dfz = np.dot(J,dz)
    return np.array([dfx,dfy,dfz],float)
if __name__ == "__main__":
   
    #file = input("choose file path\ncl")
    df = pd.read_csv("clean_data/centralised/140Hz.csv")
    times = df['timestamp']
    print(times)
    x=1; y=0; z=0
    system=np.array([x,y,z])
    derivatives = np.array([[1,0,0],[0,1,0],[0,0,1]],float)
    delta_t = 0.00998*2
    sigma = 16.0
    rho = 45.92
    beta = 4.0
    Matrix = np.eye(len(system))
    system = solve_ivp(lorentz,[times[0],times[2018]],system,t_eval=times,args = (sigma,rho,beta))
    system_n = system.y.T
    iter = 0
    exponents = []
    for t in times:
        system = system_n[iter]
        iter+=1
        Matrix_n = np.matmul(np.eye(len(system))+Jacobian(system,sigma,rho,beta)*delta_t,Matrix)
        Q,R = np.linalg.qr(Matrix_n)
        exponents.append(np.log(abs(R.diagonal())))
        Matrix = Q
    
print([sum([exponents[k][j] for k in range(2018)]) / (times[2018]) for j in range(3)])











