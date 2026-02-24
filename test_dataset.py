import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from Dataset import Dataset
from GaussianProcess import GP
from AcquisitionFunction import AF
from BayesianOptimization import BO
from Logger import Logger

def main():
    random_state = 2
    filename = "resources/grid_search_summary.csv"

    x_cols = [
        "box_threshold",
        "text_threshold"
    ]

    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="avg_time_per_sample",
        g_cols=["mAP", "energy_kwh"],
        #t_col="total_time_s",
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
                (0.2, 1.0),
                (0.0, 0.15)
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
        initial_points=5,
        random_state=random_state,
        logger=logger,
        dataset=ds,
    )

    bo.initialize()
    x_best, y_best = bo.run(n_iterations=15, verbose=True)

    print("\nBest feasible row found:")
    print("x_best =", x_best)
    print("y_best =", y_best)

    bounds_metric = "mape"
    target_metric = "accuracy"
    bo.logger.to_csv("gridsearch_results.csv", bounds_metric, target_metric)
    


if __name__ == "__main__":
    main()