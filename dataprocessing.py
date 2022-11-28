# Read BPI_Challenge_2017_1k_1.csv
import pandas as pd

import torch

bpi17 = pd.read_csv('data/BPI_Challenge_2017_1k_1.csv')

# Convert bpi17 to pkl file
dataset = {}

# Create a dictionary to map each mark to a unique integer
mark2int = {}
for i, mark in enumerate(bpi17["concept:name"].unique()):
    mark2int[mark] = i

dataset['sequences'] = []
# Extract case:concept:name
cases = bpi17["case:concept:name"].unique()
# Traverse each case and add to sequences
for case in cases:
    # Extract arrival time of each event
    arrival_time = bpi17.loc[bpi17["case:concept:name"] == case, "time:timestamp"].values
    # Convert arrival time to datetime format
    arrival_time = pd.to_datetime(arrival_time)
    # Extract start and end time
    start_time = pd.to_datetime(bpi17.loc[bpi17["case:concept:name"] == case, "time:timestamp"].min())
    end_time = pd.to_datetime(bpi17.loc[bpi17["case:concept:name"] == case, "time:timestamp"].max())
    # Convert arrival time to relative time as float type
    arrival_time = (arrival_time - start_time).total_seconds()
    # Convert arrival time Float64Index to list
    arrival_time = arrival_time.array.tolist()
    # Extract marks integer of each event
    marks = bpi17.loc[bpi17["case:concept:name"] == case, "concept:name"].values
    marks = [mark2int[mark] for mark in marks]
    # Append arrival time to sequences
    dataset['sequences'].append({
        'arrival_times': arrival_time,
        'marks': marks,
        't_start': 0,
        't_end': (end_time - start_time).total_seconds()
    })

dataset['num_marks'] = bpi17["concept:name"].unique().shape[0]

torch.save(dataset, "data/BPI_Challenge_2017_1k_1.pkl")



