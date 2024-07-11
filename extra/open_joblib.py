import joblib
import pandas as pd

# Load the joblib file
filename = 'data/hotpotqa/hotpot-qa-distractor-sample.joblib'
table_data = joblib.load(filename)

# Check if the loaded data is in a suitable format
# For instance, it should be a pandas DataFrame, or a list of dictionaries, or a similar structure

if isinstance(table_data, pd.DataFrame):
    # Directly save the DataFrame to a CSV file
    table_data.to_csv('data/hotpotqa/hotpot-qa-distractor-sample.csv', index=False)
else:
    # If the data is not already a DataFrame, convert it to one
    try:
        # Example: assuming table_data is a list of dictionaries
        df = pd.DataFrame(table_data)
        df.to_csv('data/hotpotqa/hotpot-qa-distractor-sample.csv', index=False)
    except Exception as e:
        print(f"Error converting the data to DataFrame: {e}")
