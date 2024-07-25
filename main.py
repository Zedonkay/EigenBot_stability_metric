#!/path/to/venv python3
import pandas as pd
import lyapunov_final as lyap
import numpy as np
import truncate as tr
import state_space as ss
import psd as psd

print("Running main.py")
def main():
    # Set the values for various parameters
    tau = 11
    m = 18
    delta_t = 0.01
    min_steps = 100
    force_minsteps = False
    epsilon = 10
    plotting_0=0
    plotting_final=300
    tolerance = 0.001


    # Initialize lists to store results
    exponents = []
    types = []

    # keep track of psds
    psds=[]

    # Read data from a CSV file
    df = pd.read_csv("1_raw_data/running_info.csv")
    data = df.to_numpy()

    #process each file in data

    for file in data:
        print("Running for file: ",file[0])

        #truncate data
        print("Truncating data")
        tr.main(file[0],tolerance)

        #retruncate data
        print("Retruncating data")
        tr.retruncate(file[0],file[1],file[2])
        
        # #plot state space
        # print("Plotting State Space")
        # ss.main(file[0])

        #calculate lyapunov exponents
        print("Calculating Lyapunov Exponents")
        lyap.exponent(tau,m,min_steps,epsilon,plotting_0,plotting_final,
                      delta_t, force_minsteps,types,exponents,file[0])

        #calculate psd
        print("Calculating PSD")
        psd.main(psds,file[0])

    #plot results
    print("Plotting results")
    lyap.plot_exponents(types,exponents)
    
    #plot psds
    print("Plotting PSDs")
    psd.plot_psd(types,psds)

    data = pd.DataFrame({'Type': types, 'Exponent': exponents})
    data.to_csv("3_Results/exponents.csv",index=False)

if __name__ == "__main__":
    main()
        
