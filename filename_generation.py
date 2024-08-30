def filename_raw_data(date,terrain,trial):
    return f"1_raw_data/"+str(date)+"/"+terrain+"/trial"+str(trial)+"/eigenhub_body_data.csv"

def filename_clean_data(date,terrain,trial):
    return f"2_clean_data/"+str(date)+"/"+terrain+"/trial"+str(trial)+"/"+str(date)+"_"+terrain+"_"+"trial"+str(trial)+"data.csv"

def filename_store_data(date,terrain,trial):
    return f"3_results/"+str(date)+"/"+terrain+"/trial"+str(trial)+"/"+str(date)+"_"+terrain+"_"+"trial"+str(trial)+"_"

def filename_lyapunov(date,terrain,trial):
    return filename_store_data(date,terrain,trial) + "lyapunov.csv"

def filename_big(date,terrain,trial):
    return f"3_results/"+str(date)+"/"+terrain+"/"+str(date)+"_"+terrain+"_"

