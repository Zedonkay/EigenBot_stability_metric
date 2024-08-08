
def filename_clean(terrain,object,test):
    return "1_clean_data/"+terrain+"/"+object+"/"+terrain+"_"+object+"_test"+str(test)+".csv"
def filename_raw_test(terrain,object,test):
    return "2_raw_data/"+terrain+"/"+object+"/eigenhub_"+object+"_data_"+terrain+"_"+str(test)+".csv"
def filename_lyapunov(terrain,object,test):
    return "3_results/"+terrain+"/"+object+"/test"+str(test)+"/"+object+"_test"+str(test)+"_lyapunov.csv"
def filename_exponents(terrain,object,test):
    return "3_results/"+terrain+"/"+object+"/"+terrain+"_"+object+"_exponents.csv"
def store_clean_data(terrain,object,test):
    return "3_results/"+terrain+"/"+object+"/test"+str(test)+"/"+terrain+"_"+object+"_test"+str(test)+"_"
