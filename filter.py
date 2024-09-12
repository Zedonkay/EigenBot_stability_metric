import pandas as pd

# Read the CSV files
df_to_filter = pd.read_csv("3_results/velocities.csv")
df_filter = pd.read_csv("1_raw_data/running_info.csv")

# Merge the DataFrames on 'Date', 'Terrain', and 'Trial' columns
df_merged = pd.merge(df_to_filter, df_filter, on=['Date', 'Terrain', 'Trial'], how='inner')

# Select only the columns from df_to_filter
df_filtered = df_merged[df_to_filter.columns]

# Save the filtered DataFrame to a CSV file
df_filtered.to_csv("3_results/exponents.csv", index=False)
