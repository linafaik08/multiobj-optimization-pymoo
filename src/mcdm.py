from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
import numpy as np
import plotly.graph_objects as go


def find_best_solution_by_asf(result, weights, scale=True):
    """
    Find the best solution on the Pareto front using Augmented Scalarization Function (ASF).

    Parameters
    ----------
    result : pymoo.optimize.MinimizationResult
        Optimization result from pymoo containing F and X.
    weights : list or np.ndarray
        Preference weights for each objective (higher = more important).
    scale : bool
        Whether to scale F between 0-1 (based on ideal/nadir). Recommended.

    Returns
    -------
    idx : int
        Index of the selected solution.
    F_best : np.ndarray
        Objective values of the selected solution.
    """
    F = result.F
    if scale:
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)
        nF = (F - ideal) / (nadir - ideal)
    else:
        nF = F

    decomp = ASF()
    idx = decomp.do(nF, 1 / np.array(weights)).argmin()
    return idx, result.F[idx]


def find_best_solution_by_pseudo_weights(result, weights, scale=True):
    """
    Find the best solution on the Pareto front using the Pseudo Weights method.

    Parameters
    ----------
    result : pymoo.optimize.MinimizationResult
        Optimization result from pymoo containing F and X.
    weights : list or np.ndarray
        Preference weights for each objective (higher = more important).
    scale : bool
        Whether to scale F between 0-1 (based on ideal/nadir). Recommended.

    Returns
    -------
    idx : int
        Index of the selected solution.
    F_best : np.ndarray
        Objective values of the selected solution.
    """
    F = result.F
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    nF = (F - ideal) / (nadir - ideal) if scale else F

    idx = PseudoWeights(weights).do(nF)
    return idx, result.F[idx]