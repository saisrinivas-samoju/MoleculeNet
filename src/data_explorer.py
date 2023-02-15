from typing import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_loader import MoleculeDataset


def make_df_for_eda(dataset: MoleculeDataset) -> pd.DataFrame:
    """Make a dataframe for EDA from a MoleculeDataset.

    Args:
        dataset (MoleculeDataset): MoleculeDataset

    Returns:
        pd.DataFrame: A dataframe for EDA
    """
    df = pd.DataFrame([
        {
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "label": data.y
        }
        for data in dataset
    ])

    sample_y = df.iloc[0]['label']

    if isinstance(sample_y, Sequence):
        for i in range(len(sample_y)):
            df[f'label_{i+1}'] = df['label'].apply(lambda x: x[i])
        df = df.drop(columns=['label'])
        
    return df



def plot_histograms(data_list: list[dict]) -> go.Figure:
    """
    Create plotly histograms from a list of dictionaries.
    
    Parameters:
    -----------
    data_list : list of dict
        Each dictionary must contain:
        - 'data': array-like data to plot
        - 'title': title of the histogram
        - 'x_label': label for x-axis
        - 'y_label': label for y-axis
        - 'n_bins': number of bins for histogram
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure with the histograms
    """
    # Create subplot figure with as many rows as histograms
    fig = make_subplots(rows=len(data_list), cols=1, 
                        subplot_titles=[item['title'] for item in data_list])
    
    # Add each histogram to the figure
    for i, item in enumerate(data_list):
        # Create histogram
        hist = go.Histogram(
            x=item['data'],
            nbinsx=item['n_bins'],
            name=item['title']
        )
        
        # Add trace to the subplot
        fig.add_trace(hist, row=i+1, col=1)
        
        # Update axes labels
        fig.update_xaxes(title_text=item['x_label'], row=i+1, col=1)
        fig.update_yaxes(title_text=item['y_label'], row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=300*len(data_list),
        width=800,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig

def explore_with_histograms(dataset: MoleculeDataset) -> go.Figure:
    """Explore a MoleculeDataset with histograms.

    Args:
        dataset (MoleculeDataset): MoleculeDataset

    Returns:
        go.Figure: A plotly figure with the histograms
    """
    df = make_df_for_eda(dataset)
    hist_data = [
        {
            "data": df[col],
            "title": f"Distribution of {col}",
            "x_label": col,
            "y_label": "Count",
            "n_bins": 30
        } for col in df.columns
    ]
    fig = plot_histograms(hist_data)
    # fig.show()

    return fig