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
    pass


def plot_histograms(data_list: list[dict]) -> go.Figure:
    """
    Create plotly histograms from a list of dictionaries.
    
    Parameters:
    -----------
    data_list : list of data dict
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure with the histograms
    """
    pass

def explore_with_histograms(dataset: MoleculeDataset) -> go.Figure:
    """Explore a MoleculeDataset with histograms.

    Args:
        dataset (MoleculeDataset): MoleculeDataset

    Returns:
        go.Figure: A plotly figure with the histograms
    """
    pass