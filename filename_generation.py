def filename_raw_retest(frequency = int, test=str, control_type=str):
    if(control_type == "centralised"):
        return "2_raw_data/official_tests/"+control_type+"/retest_curve/"+"odometry_data_centralised_retest_"+str(frequency)+"Hz.csv"
    else:
        return "2_raw_data/official_tests/"+control_type+"/odometry_data_distributed_retest_"+str(frequency)+"Hz.csv"
def filename_clean(frequency = int, test=str, control_type=str):
    return "1_clean_data/"+control_type+"/"+str(frequency)+"hz/"+"truncated_"+str(frequency)+"hz_test"+test+".csv"
def filename_raw_test(frequency=int,test=str,control_type=str):
    return "2_raw_data/official_tests/"+control_type+"/odometry_data_"+control_type+"_test_"+str(frequency)+"hz_"+test+".csv"

def raw_predefined_v_neural(frequency = int, test=str, control_type=str):
    return "3_raw_predefined_v_neural/"+control_type+"/"+"pos_quart_"+test+control_type+".csv"

def four_sample_data(frequency = int, test=str, control_type=str):
    return "4_sample_data/"+control_type+"/"+str(frequency)+"hz_"+test+".csv"

def store_clean_data(frequency = int, test=str, control_type=str):
    return "6_results/clean_data/"+control_type+"/"+str(frequency)+'hz/'+'test'+test+"/"+str(frequency)+"hz_"+'test'+test+"_"

def store_raw_retest(frequency = int, test=str, control_type=str):
    return "6_results/raw_data/"+control_type+"/retest/"+str(frequency)+"hz/"+str(frequency)+"Hz_"

def store_raw_test(frequency = int, test=str, control_type=str):
    return "6_results/raw_data/"+control_type+"/"+str(frequency)+"hz/"+"test"+test+"/"+str(frequency)+"Hz_"+test+"_"
