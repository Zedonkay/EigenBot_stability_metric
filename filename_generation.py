def filename_raw_retest(frequency = int, test=int, control_type=str):
    if(control_type == "centralised"):
        return "2_raw_data/official_tests/"+control_type+"/retest_curve/"+"odometry_data_centralised_retest_"+str(frequency)+"Hz.csv"
    else:
        return "2_raw_data/official_tests/"+control_type+"/odometry_data_distributed_retest_"+str(frequency)+"Hz.csv"
def filename_clean(frequency = int, test=int, control_type=str):
    return "1_clean_data/"+control_type+"/"+str(frequency)+"hz/"+control_type+"_"+str(frequency)+"hz_test"+str(test)+".csv"
def filename_raw_test(frequency=int,test=int,control_type=str):
    return "2_raw_data/official_tests/"+control_type+"/odometry_data_"+control_type+"_test_"+str(frequency)+"hz_"+str(test)+".csv"
def filename_lyapunov(frequency=int, test=int, control_type=str):
    return "6_results/clean_data/"+control_type+"/"+str(frequency)+"hz/"+"test"+str(test)+"/"+control_type+"_"+str(frequency)+"hz_test"+str(test)+"_lyapunovdata.csv"
def filename_exponents(frequency=int, test=int, control_type=str):
    return "6_results/clean_data/"+control_type+"/"+control_type+"_exponents.csv"
def raw_predefined_v_neural(frequency = int, test=int, control_type=str):
    return "3_raw_predefined_v_neural/"+control_type+"/"+"pos_quart_"+str(test)+control_type+".csv"

def four_sample_data(frequency = int, test=int, control_type=str):
    return "4_sample_data/"+control_type+"/"+str(frequency)+"hz_"+str(test)+".csv"

def store_clean_data(frequency = int, test=int, control_type=str):
    return "6_results/clean_data/"+control_type+"/"+str(frequency)+'hz/'+'test'+str(test)+"/"+control_type+"_"+str(frequency)+"hz_"+'test'+str(test)+"_"

def store_raw_retest(frequency = int, test=int, control_type=str):
    return "6_results/raw_data/"+control_type+"/retest/"+str(frequency)+"hz/"+str(frequency)+"Hz_"

def store_raw_test(frequency = int, test=int, control_type=str):
    return "6_results/raw_data/"+control_type+"/"+str(frequency)+"hz/"+"test"+str(test)+"/"+str(frequency)+"Hz_"+str(test)+"_"
