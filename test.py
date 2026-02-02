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


    X0 = np.array([[1.0,-0.7091799501804994,-4.156684027273989,1.0,1.244252553018157],
                    [-5.0,3.836235550162306,3.625467202835888,0.0,3.0967906597468593],
                    [-3.0,2.0031888885616036,-3.532797390175573,3.0,4.696410430099664],
                    [5.0,1.6721435340103712,-3.444857217820294,2.0,-2.2733290225162093],
                    [5.0,-2.054706549762683,0.6254284259976153,2.0,-3.6994298798002423],
                    [-1.0,2.7588101512060357,-0.884742579564846,3.0,0.8381836953216251],
                    [-5.0,2.354020946397921,1.781514786574962,3.0,-3.594941900954508],
                    [5.0,3.7647723272948435,4.1274352815349795,4.0,1.240770644474452],
                    [-1.0,-4.526434606391648,2.0335966205144267,3.0,1.5261371722839918],
                    [-5.0,4.563793985988611,-1.1565955690592058,0.0,5.009438447162128],
                    [3.0,0.2375818300256718,2.7099917333634833,3.0,1.7064228325333577],
                    [-1.0,-1.8526606017932532,1.737123829931745,1.0,-3.0585484731888095],
                    [-3.0,-0.4070681304069179,1.1705940170928795,2.0,2.7676684868292334],
                    [5.0,-4.212095592576338,-3.472448989779326,3.0,-1.508421859194149],
                    [1.0,-4.3448361303044605,-3.6791068096629966,4.0,-0.6786326375414591],
                    [-3.0,4.3517264333650365,-2.7654424396322863,0.0,0.3895020496654702],
                    [1.0,-4.266621517874105,1.8113982871725742,1.0,5.037877999686546],
                    [-1.0,2.2159062311508766,1.3028113940712798,2.0,-4.532919105878091],
                    [3.0,-1.8448059271009196,0.3860946045172762,3.0,1.7151900641635178],
                    [-3.0,4.988066925951791,3.0508707146474032,2.0,2.5264571886236693]])

    #bo.initialize(X0, objective(X0), np.column_stack([g1(X0), g2(X0)]))
    bo.initialize()
    x_best, y_best = bo.run(n_iterations=40, n_restarts=10, verbose=True)

    print("\nBest feasible solution found:")
    print("x_best =", x_best)
    print("y_best =", y_best)

    bo.logger.to_csv("rastrigin_mixed_log.csv", bounds_metric="mape", target_metric="accuracy")


if __name__ == "__main__":
    main()
