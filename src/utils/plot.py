from typing import Dict

import numpy as np
import plotly.graph_objects as go


def plot(x: np.ndarray, y: Dict[str, np.ndarray], title: str, xaxis: str, yaxis: str):
    """
    Plots a line chart with multiple data series.

    :param x: Numpy array for the x-axis values.
    :param y: Dictionary where keys are series names and values are numpy arrays for the y-axis.
    :param title: Title of the plot.
    :param xaxis: Label for the x-axis.
    :param yaxis: Label for the y-axis.
    """
    fig = go.Figure()

    for name, y_data in y.items():
        fig.add_trace(go.Scatter(x=x, y=y_data, mode="lines", name=name))

    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()
