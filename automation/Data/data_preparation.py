import os
import pandas as pd
from datetime import datetime, timedelta

def prepare_dataset():

    # Calculate the dates for the last 3 days
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]
    
    

    # Generate file names based on the dates
    file_names = [f'kafka_data_{date}.csv' for date in dates]
    print(file_names)

    # Initialize a list to hold the DataFrames
    dataframes = []

    # Read each CSV file into a DataFrame and append it to the list
    for file_name in file_names:
    
        try:
            df = pd.read_csv(file_name)
            os.remove(file_name)
            print(f"{file_name} has been deleted.")
            dataframes.append(df)
        except FileNotFoundError:
            print(f'File not found: {file_name}')

    # Concatenate all DataFrames in the list into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    merged_file_name = 'kafka_data.csv'
    merged_file_path=os.path.join("/home/team18/deploy/", merged_file_name)
    merged_df.to_csv(merged_file_path, index=False)
    
    print(os.path.abspath(merged_file_path))
    print(f'Merged data saved to {merged_file_path}.')

if __name__ == '__main__':
    prepare_dataset()
