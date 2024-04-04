import os
import pandas as pd

folder_path = "reddit-dataset-master"
dataframes = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df['file_name'] = file_name  # Add a new column with the file name
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
