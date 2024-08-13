
def filename_clean(control_type,terrain,leg):
    return "1_clean_data/"+control_type+"/"+terrain+"/"+terrain+control_type+"_leg"+str(leg)+".csv"
def filename_raw_test(terrain,control_type,test):
    return "2_raw_data/"+control_type+"/"+"contact_data_"+terrain+"_"+control_type+".csv"
def filename_raw_legs(control_type,terrain,leg):
    return "2_raw_data/"+control_type+"/"+terrain+""+control_type+"_leg"+str(leg)+".csv"
def filename_lyapunov(control_type,terrain,leg):
    return store_clean_data(control_type,terrain,leg)+"lyapunovdata.csv"
def store_clean_data(control_type,terrain,leg):
    return "3_results/"+control_type+"/"+terrain+"/leg"+str(leg)+"/"+terrain+"_"+control_type+"_leg"+str(leg)+"_"