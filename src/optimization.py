import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
np.random.seed(42)

import numpy as np
from pymoo.core.problem import ElementwiseProblem

class DispatchProblem(ElementwiseProblem):
    def __init__(
        self, 
        data,
        column_demand,
        column_stock,
        **kwargs
        ):
        """
        Multi-objective dispatch problem for product allocation with optional demand uncertainty.

        Objectives:
            - Maximize total sales (minimize negative expected sales)
            - Minimize overstock (expected unsold units)

        Constraint:
            - Total dispatched units must not exceed total available stock.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing dispatch targets (rows per store).
        column_demand : str
            The column name representing forecasted demand per store.
        column_stock : str
            The column name representing available stock (total across all stores).
        nb_scenarios : int, optional
            Number of demand scenarios to simulate for uncertainty (default: 20).
        std_forecast_error : float, optional
            Relative standard deviation of forecast error (default: 0.1 = 10%).
        """
        
        self.data = data
        self.demand = data[column_demand].values
        self.total_available = data[column_stock].values[0]
        
        self.with_demand_uncertainty = kwargs.get("with_demand_uncertainty", False)
        self.nb_scenarios = kwargs.get("nb_scenarios", 20)
        self.std_forecast_error = kwargs.get("nb_scenarios", 0.1)

        super().__init__(n_var=len(data),
                         n_obj=2, # number of objectives
                         n_constr=1, # number of constraints
                         xl=np.zeros(len(data)), # lower bounds
                         xu=self.total_available # upper bounds
                         ) 
        
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Perform the actual evaluation of the objective and constraint functions
        for a single solution vector `x`.

        The majority of optimization algorithms in pymoo are population-based,
        which means multiple candidate solutions are evaluated per generation.
        The `_evaluate` method is called to compute the objectives and constraints
        for each solution, and store them in the `out` dictionary.

        Expected keys in `out`:
            - out["F"]: objective values (array-like)
            - out["G"]: constraint violations (only if n_constr > 0)

        Parameters:
            x : np.ndarray
                A single candidate solution vector (e.g., dispatch quantities for each product-store)
            out : dict
                Dictionary used by pymoo to collect the evaluation results.
            *args, **kwargs :
                Optional additional arguments — typically not needed unless
                you're passing extra dynamic information (e.g., surrogate models,
                problem metadata, batch IDs).
        """
        
        if not(self.with_demand_uncertainty):
            self._evaluate_simple(x, out)
        else:
            self._evaluate_with_demand_uncertainty(x, out)

    def _evaluate_simple(self, x, out):
        """
        Deterministic evaluation with exact forecast.
        """
        
        # Objective 1: Maximize sales (pymoo minimizes, so we negate it)
        
        sales = np.minimum(x, self.demand)
        f1 = -np.sum(sales)
       
        # Objective 2: Minimize overstock (unsold units)
        overstock = np.maximum(x - self.demand, 0)
        f2 = np.sum(overstock)

        # Constraint: total dispatch must not exceed total available units
        g1 = np.sum(x) - self.total_available

        out["F"] = [f1, f2]
        out["G"] = [g1]
        
    def _evaluate_with_demand_uncertainty(self, x, out):
        """
        Stochastic evaluation under uncertain demand (Monte Carlo sampling).
        """
        
        scenarios = np.random.normal(
            loc=self.demand, 
            scale=self.std_forecast_error * self.demand,
            size=(self.nb_scenarios, len(x))
            )
        sales_per_scenario = np.minimum(x[None, :], scenarios)
        expected_sales = np.mean(np.sum(sales_per_scenario, axis=1))
        
        f1 = -expected_sales
        f2 = np.mean(np.sum(np.maximum(x[None, :] - scenarios, 0), axis=1))  # Expected overstock

        # Constraint: total dispatch must not exceed total available units
        g1 = np.sum(x) - self.total_available

        out["F"] = [f1, f2]
        out["G"] = [g1]


def run_dispatch_optimization(
    problem,
    algorithm=None,
    n_gen=100,
    seed=42,
    verbose=True,
    **kwargs
):
    """
    Runs multi-objective optimization on a dispatch problem using a customizable algorithm.

    Parameters
    ----------
    problem : pymoo.core.problem.ElementwiseProblem
        A pymoo-compatible problem instance to be solved.
    algorithm : pymoo algorithm instance, optional
        If None, defaults to NSGA-II with standard settings.
    n_gen : int, default=100
        Number of generations (iterations) to run the optimization.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to show progress during optimization.
    **kwargs : dict
        Additional parameters passed to pymoo's `minimize()` function.

    Returns
    -------
    result : pymoo.optimize.MinimizationResult
        The result of the optimization, containing Pareto front, solutions, and history.
    """

    if algorithm is None:
        algorithm = NSGA2(
            pop_size=100,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
    termination = ("n_gen", n_gen)
    if "termination" in kwargs:
        termination = kwargs.pop("termination")
        
        
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        **kwargs
    )
        
    return result

