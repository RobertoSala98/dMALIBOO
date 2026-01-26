import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from GaussianProcess import GP
from AcquisitionFunction import AF
from BayesianOptimization import BO
from Logger import Logger


def rastrigin(X, A=10.0):
    """
    Standard Rastrigin (min at 0).
    Here BO minimizes y, so this is fine.
    """
    X = np.asarray(X)
    n = X.shape[1]
    return A * n + np.sum(X**2 - A * np.cos(2.0 * np.pi * X), axis=1)


def objective(X):
    """
    Mixed variables in 5D:
      x0 discrete in {-5,-3,-1,1,3,5}
      x1 continuous in [-5.12, 5.12]
      x2 continuous in [-5.12, 5.12]
      x3 discrete in {0,1,2,3,4}
      x4 continuous in [-5.12, 5.12]
    """
    X = np.asarray(X)
    return rastrigin(X)


def g1(X):
    # "budget" type: keep sum in [-2, 2]
    return np.sum(X, axis=1)


def g2(X):
    # keep a nonlinear expression in [0, 15]
    return (X[:, 0] ** 2) + (X[:, 1] * X[:, 2]) + 2.0 * X[:, 3] + np.abs(X[:, 4])


def main():
    random_state = 16

    bounds = np.array([
        [-5.12, 5.12],  # x0 (discretized)
        [-5.12, 5.12],  # x1
        [-5.12, 5.12],  # x2
        [0.0, 4.0],     # x3 (discretized)
        [-5.12, 5.12],  # x4
    ])

    # Mixed-variable spec: only x0 and x3 are discrete
    discrete_values = [
        np.array([-5, -3, -1, 1, 3, 5], dtype=float),  # x0
        None,                                          # x1
        None,                                          # x2
        np.arange(0.0, 5.0, 1.0),                       # x3 in {0,1,2,3,4}
        None,                                          # x4
    ]

    gp = GP(kernel_name="Matern", random_state=random_state, length_scale=1.0)()

    af = AF(
        kind="ei",
        kappa=1.0,

        ml_on_bounds=True,
        ml_on_bounds_parameters={
            "name": "ridge",
            "task": "regression",
            "constraint_bounds": [
                (-2.0, 2.0),    # g1 bounds
                (0.0, 15.0),    # g2 bounds
            ],
        },

        ml_on_target=True,
        ml_on_target_parameters={
            "name": "ridge",
            "task": "classification",
        },

        bounds=bounds,
        random_state=random_state,
    )

    logger = Logger(dim=bounds.shape[0])

    bo = BO(
        objective_function=objective,
        domain_bounds=bounds,
        gp=gp,
        af=af,
        constraint_functions=[g1, g2],
        initial_points=20,
        random_state=random_state,
        logger=logger,

        discrete_values=discrete_values,
        discrete_refine=True,

        epsilon_greedy = 0.10,
        epsilon_max_tries = 1000,
        epsilon_feas_threshold = 0.8,
    )

    bo.initialize()
    x_best, y_best = bo.run(n_iterations=40, n_restarts=10, verbose=True)

    print("\nBest feasible solution found:")
    print("x_best =", x_best)
    print("y_best =", y_best)

    bo.logger.to_csv("rastrigin_mixed_log.csv", bounds_metric="mape", target_metric="accuracy")


if __name__ == "__main__":
    main()
