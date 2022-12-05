import pandas as pd
import plotly.express as px


def visualize(d_train, d_val, d_test):
    """
    Visualize the split dataset.
    Args:
        d_train: training dataset
        d_val: validation dataset
        d_test: test dataset
    """
    df = pd.DataFrame([])
    for sequence in d_train.sequences:
        df = df.append({'start_time': pd.to_datetime(sequence['t_start'], unit='s'),
                        'end_time': pd.to_datetime(sequence['t_end'], unit='s'), 'case': sequence['case'],
                        'set': 'train'},
                       ignore_index=True)
    for sequence in d_val.sequences:
        df = df.append({'start_time': pd.to_datetime(sequence['t_start'], unit='s'),
                        'end_time': pd.to_datetime(sequence['t_end'], unit='s'), 'case': sequence['case'],
                        'set': 'val'},
                       ignore_index=True)
    for sequence in d_test.sequences:
        df = df.append({'start_time': pd.to_datetime(sequence['t_start'], unit='s'),
                        'end_time': pd.to_datetime(sequence['t_end'], unit='s'), 'case': sequence['case'],
                        'set': 'test'},
                       ignore_index=True)
    fig = px.timeline(df, x_start='start_time', x_end='end_time', y='case', color='set', title='Splitting dataset')
    fig.show()
