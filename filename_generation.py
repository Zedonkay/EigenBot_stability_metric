
def filename_clean(disturbance,control_type):
    return "1_clean_data/"+disturbance+"/"+disturbance+"_"+control_type+".csv"
def filename_raw_test(disturbance,control_type):
    return "2_raw_data/"+disturbance+"/"+"pos_quart_"+disturbance+"_"+control_type+".csv"
def filename_lyapunov(disturbance,control_type):
    return "3_results/"+disturbance+"/"+control_type+"/"+disturbance+"_"+control_type+"_"+"_lyapunov.csv"
def filename_exponents(disturbance,control_type):
    return "3_results/"+disturbance+"/"+disturbance+"_exponents.csv"
def store_clean_data(disturbance,control_type):
    return "3_results/"+disturbance+"/"+control_type+"/"+disturbance+"_"+control_type+"_"

