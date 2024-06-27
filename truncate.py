import pandas as pd
import numpy as np
import filename_generation as fg

def find_start_and_end(data,tolerance):
    start = find_start(data,tolerance)
    end = find_end(start,data,tolerance)
    return start,end
    
def find_start(data,tolerance):
    ret = []
    for i in range(len(data)-1):
        if np.linalg.norm(data[i]-data[i+1]>tolerance):
            return i
        else:
           if(i<500):
               ret.append(np.linalg.norm(data[i]-data[i+1]))
    return 0

        
def find_end(start,data,tolerance):
    for i in range(len(data)-1,0,-1):
        if np.linalg.norm(data[i]-data[i-1]>tolerance):
            return i
        
def main(frequency,test,control_type,tolerance):
    filename = fg.filename_raw_test(frequency,test,control_type)
    df = pd.read_csv(filename)
    data = df[['pz']].values
    start,end = find_start_and_end(data,tolerance)
    df = df[start:end]
    df.to_csv(fg.filename_clean(frequency,test,control_type),index=False)


if __name__ == '__main__':
    distributed =[[100,'1'],[100,'2'],[100,'3'],[140,'1'],[140,'2'],[140,'3'],
                  [180,'1'],[180,'2'],[180,'3'],[220,'1'],[220,'2'],[220,'3'],
                  [260,'1'],[260,'2'],[260,'3'],[300,'1'],[300,'2'],[300,'3'],
                  [350,'1'],[350,'2'],[350,'3'],[370,'1'],[370,'2'],[370,'3'],[400,'1'],
                  [400,'2'],[400,'3'],[500,'1'],[600,'1'],[700,'1'],[800,'1'],
                  [1000,'1'],[1040,'1'],[1080,'1'],[1120,'1'],[1160,'1'],[1200,'1'],
                  [1400,'1'],[1440,'1'],[1480,'1'],[1600,'1'], [1800,'1'],[2000,'1']]
    
    centralised = [[100,'1'],[100,'2'], [100,'3'],[140,'1'],[140,'2'],
                   [140,'3'], [140,'4'],[180,'1'],[220,'1'],[220,'2'],
                   [220,'3'],[220,'4'],[260,'1'],[260,'2'],[260,'3'],
                   [260,'4'],[300,'1'],[320,'1'],[320,'2'],[320,'3'],
                   [330,'1'],[350,'1'],[350,'2'],[350,'3'],[350,'4'],
                   [370,'1'],[370,'2'],[370,'3']]
    tolerance = 0.001
    
    print("running centralised")
    for i in centralised:
        main(i[0],i[1],'centralised',tolerance)
    print("running distributed")
    for i in distributed:
        main(i[0],i[1],'distributed',tolerance)
    print("done")