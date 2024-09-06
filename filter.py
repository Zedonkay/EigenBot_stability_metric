import pandas as pd

df_to_filter = pd.read_csv("3_results/exponents.csv")
df_filter =pd.read_csv("1_raw_data/running_info.csv")
df_filtered = df_to_filter[df_to_filter[['Date', 'Terrain', 'Trial']].isin(df_filter[['Date', 'Terrain', 'Trial']]).all(axis=1)]
df_to_filter.to_csv("3_results/velocities.csv", index=False)  # Save the DataFrame to a CSV file
df_filtered.to_csv("3_results/exponents.csv", index=False)  # Save the DataFrame to a CSV file