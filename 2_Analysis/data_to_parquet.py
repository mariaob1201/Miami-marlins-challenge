import pandas as pd

# Read CSV
df = pd.read_csv("1_Data/exit_velo_project_data.csv")

# Save as Parquet
#df.to_parquet("1_Data/exit_velo_project_data.parquet", engine="pyarrow", index=False)

# Save as Parquet (fastparquet engine)
df.to_parquet("1_Data/exit_velo_project_data_fp.parquet", engine="fastparquet", index=False)
