import nolds
import numpy as np
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("clean_data/centralised/140Hz.csv")
    data = df[['px']].values
    data= np.reshape(np.array(data), (1,len(data)))[0]
    print(data)
    lyapunov = nolds.lyap_r(data)
    print(lyapunov)