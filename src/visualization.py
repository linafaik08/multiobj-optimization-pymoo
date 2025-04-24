import plotly.graph_objects as go
import numpy as np

def plot_objective_space(
    result,
    x_label="Objective 1",
    y_label="Objective 2",
    title="Objective Space",
    show_ideal_nadir=True,
    scale_objectives=False,
    highlight_points=None,
):
    """
    Plot the Pareto front in the objective space using Plotly, with optional scaling and reference points.

    Parameters
    ----------
    result : pymoo.optimize.MinimizationResult
        The result object returned by pymoo's minimize().
    x_label : str, optional
        Label for the x-axis (default is "Objective 1").
    y_label : str, optional
        Label for the y-axis (default is "Objective 2").
    title : str, optional
        Title of the plot (default is "Objective Space").
    show_ideal_nadir : bool, optional
        Whether to show approximate ideal and nadir points (default: True).
    scale_objectives : bool, optional
        Whether to scale objectives using min-max normalization (default: False).
    highlight_points : list of dicts, optional
        List of points to highlight. Each dict should have:
            - "index" : int (index of the point in result.F)
            - "label" : str (label to show in legend)
            - "color" : str (color of the marker)
    """
    F = result.F

    if scale_objectives:
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        F = (F - approx_ideal) / (approx_nadir - approx_ideal)
        
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=F[:, 0],
        y=F[:, 1],
        mode='markers',
        marker=dict(size=8, color='blue', line=dict(width=1)),
        name="Pareto Front"
    ))

    if show_ideal_nadir:
        fig.add_trace(go.Scatter(
            x=[approx_ideal[0]],
            y=[approx_ideal[1]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='star'),
            name="Ideal Point (Approx)"
        ))

        fig.add_trace(go.Scatter(
            x=[approx_nadir[0]],
            y=[approx_nadir[1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='pentagon'),
            name="Nadir Point (Approx)"
        ))
        
    if highlight_points is not None:
        for highlight_point in highlight_points:
            fig.add_trace(go.Scatter(
                x=[F[highlight_point["index"], 0]],
                y=[F[highlight_point["index"], 1]],
                mode='markers',
                marker=dict(size=14, color=highlight_point["color"], symbol='x'),
                name=highlight_point["label"]
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label + (" (scaled)" if scale_objectives else ""),
        yaxis_title=y_label + (" (scaled)" if scale_objectives else ""),
        template="plotly_white",
        width=700,
        height=500
    )

    return fig