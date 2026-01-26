import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from Dataset import Dataset
from GaussianProcess import GP
from AcquisitionFunction import AF
from BayesianOptimization import BO
from Logger import Logger

def main():
    random_state = 16
    filename = "resources/ligen.csv"

    x_cols = [
        "ALIGN_SPLIT",
        "OPTIMIZE_SPLIT",
        "OPTIMIZE_REPS",
        "CUDA_THREADS",
        "N_RESTART",
        "CLIPPING",
        "SIM_THRESH",
        "BUFFER_SIZE"
    ]

    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="RMSD^3*TIME",
        g_cols=["RMSD_0.75"],
        t_col="TIME_TOTAL",
    )

    bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])

    gp = GP(kernel_name="Matern", random_state=random_state, length_scale=1.0)()

    af = AF(
        kind="ei",
        kappa=1.0,

        ml_on_bounds=True,
        ml_on_bounds_parameters={
            "name": "ridge",
            "task": "regression",
            "constraint_bounds": [
                (0.0, 2.1)
            ],
        },

        ml_on_target=True,
        ml_on_target_parameters={
            "name": "nn",
            "task": "classification",
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
        },

        bounds=bounds,
        random_state=random_state,
    )

    logger = Logger(dim=ds.dim)

    def dummy_objective(_X):
        raise RuntimeError("Objective should not be called in dataset mode.")

    bo = BO(
        objective_function=dummy_objective,
        domain_bounds=bounds,
        gp=gp,
        af=af,
        constraint_functions=None,
        initial_points=12,
        random_state=random_state,
        logger=logger,
        dataset=ds,
    )

    bo.initialize()
    x_best, y_best = bo.run(n_iterations=60, n_restarts=5, verbose=True)

    print("\nBest feasible row found:")
    print("x_best =", x_best)
    print("y_best =", y_best)

    bounds_metric = "mape"
    target_metric = "accuracy"
    bo.logger.to_csv("test_dataset_log.csv", bounds_metric, target_metric)
    


if __name__ == "__main__":
    main()