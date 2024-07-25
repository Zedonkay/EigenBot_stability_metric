def filename_raw_data(type):
    return f"1_raw_data/eigenhub_{type}_data.csv"

def filename_clean_data(type):
    return f"2_clean_data/{type}_cleaned_data.csv"

def filename_store_data(type):
    return f"3_results/{type}/{type}_"

def filename_lyapunov(type):
    return f"3_results/{type}/{type}_lyapunov.csv"

def filename_exponents():
    return "3_results/exponents.csv"