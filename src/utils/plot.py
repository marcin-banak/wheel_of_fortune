from typing import Dict

import numpy as np
import plotly.graph_objects as go


def plot(x: np.ndarray, y: Dict[str, np.ndarray], title: str, xaxis: str, yaxis: str):
    fig = go.Figure()

    for name, y_data in y.items():
        fig.add_trace(go.Scatter(x=x, y=y_data, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()
