import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pathlib import Path

from Dataset import Dataset
from GaussianProcess import GP
from AcquisitionFunction import AF
from BayesianOptimization import BO
from Logger import Logger


def load_initial_point_indices(
    init_points_csv,
    dataset,
    feature_columns,
    n_initial_points,
):
    """
    Load the first n_initial_points rows from the initial-points CSV.

    Each row is matched to exactly one dataset row using both the
    feature columns and the target column. Returns dataset row indices.
    """
    if not os.path.exists(init_points_csv):
        raise FileNotFoundError(
            f"Initial-points CSV not found: {init_points_csv}"
        )

    init_points = pd.read_csv(init_points_csv)
    target_column_logs = 'y'

    # Use only the first n_initial_points rows.
    init_points = init_points.head(n_initial_points)

    required_columns = list(feature_columns) + [target_column_logs]
    missing_columns = [
        column
        for column in required_columns
        if column not in init_points.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Initial-points CSV is missing required columns: "
            f"{missing_columns}"
        )

    matched_indices = []
    seen_configurations = set()

    for _, init_row in init_points.iterrows():
        mask = np.ones(dataset.n, dtype=bool)

        for column in feature_columns:
            feature_index = dataset.x_cols.index(column)

            mask &= np.isclose(
                dataset.X[:, feature_index],
                float(init_row[column]),
                rtol=1e-9,
                atol=1e-12,
            )

        mask &= np.isclose(
            dataset.y,
            float(init_row[target_column_logs]),
            rtol=1e-9,
            atol=1e-12,
        )

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) != 1:
            print(
                "Skipping initial point: no unique dataset match for",
                {
                    column: init_row[column]
                    for column in required_columns
                },
            )
            continue

        idx = int(candidate_indices[0])
        configuration = tuple(dataset.X[idx])

        if configuration in seen_configurations:
            print(
                "Skipping duplicate initial-point configuration:",
                configuration,
            )
            continue

        seen_configurations.add(configuration)
        matched_indices.append(idx)

    if not matched_indices:
        raise ValueError(
            "No initial point could be uniquely matched to the dataset."
        )

    print(
        f"Loaded {len(matched_indices)} matched initial points from "
        f"{init_points_csv}"
    )

    return np.asarray(matched_indices, dtype=int)


def main():

    INPUT_FOLDER = 'inputs_boPR'

    # ### OSCAR-P ###
    path = "output_with_initial_points_DiscreteBO/oscarp"
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    n_init_points = 3

    filename = "resources/oscarp.csv"
    x_cols = [
        "parallelism_ffmpeg-0",
        "parallelism_librosa",
        "parallelism_ffmpeg-1",
        "parallelism_ffmpeg-2",
        "parallelism_deepspeech"
    ]
    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="cost",
        g_cols=["total_time"],
        t_col="total_time",
    )
    bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])

    constr_values = [150, 300, 450, 600]

    for constr in constr_values:

        path = f"output_with_initial_points_DiscreteBO/oscarp/{constr}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        bo_PR_logs_folder = os.path.join(INPUT_FOLDER, 'oscarp', str(constr))

        csv_files = [
                file for file in os.listdir(bo_PR_logs_folder)
                if file.lower().endswith(".csv")
            ]

        for file in csv_files:
            seed = int(file.split(".csv")[0])

            init_points_csv = os.path.join(bo_PR_logs_folder, file)

            initial_indices = load_initial_point_indices(
                init_points_csv=init_points_csv,
                dataset=ds,
                feature_columns=ds.x_cols,
                n_initial_points=n_init_points,
            )

            gp_builder = GP(kernel_name="RBF", 
                random_state=seed, 
                length_scale=1.0)

            af = AF(
                kind="lcb",
                kappa=1.0,
                ml_on_bounds=False,
                ml_on_bounds_parameters={
                    "constraint_bounds": [[0, constr]],
                },
                ml_on_target=False,
                bounds=bounds,
                random_state=seed,
            )

            logger = Logger(dim=ds.dim)

            def dummy_objective(X):
                raise RuntimeError("Objective should not be called in dataset mode.")

            bo = BO(
                objective_function=dummy_objective,
                domain_bounds=bounds,
                gp_builder=gp_builder,
                af=af,
                constraint_functions=[],
                initial_points=3,
                init_strategy="random",
                random_state=seed,
                logger=logger,
                dataset=ds,
                epsilon_greedy=0.0,
                reparameterization=True
            )

            print(f"{path}/{seed}.csv")

            bo.initialize(X0=initial_indices)

            x_best, y_best = bo.run(n_iterations=10, n_restarts=10, verbose=True)
            print("Best feasible x =", x_best, " y =", y_best)

            bounds_metric = "mape"
            target_metric = "accuracy"
            bo.logger.to_csv(f"{path}/{seed}.csv", bounds_metric, target_metric)


    ### Query52 ###
    path = "output_with_initial_points_DiscreteBO/query52"
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    n_init_points = 3

    filename = "resources/query52_vm_ram.csv"
    x_cols = ["#vm","ram"]

    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="cost",
        g_cols=["time"],
        t_col="time",
    )
    bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])

    constr_values = [262887, 334000, 407000, 645000, 2250000]

    for constr in constr_values:

        path = f"output_with_initial_points_DiscreteBO/query52/{constr}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        bo_PR_logs_folder = os.path.join(INPUT_FOLDER, 'query52', str(constr))

        csv_files = [
                file for file in os.listdir(bo_PR_logs_folder)
                if file.lower().endswith(".csv")
            ]

        for file in csv_files:
            seed = int(file.split(".csv")[0])

            init_points_csv = os.path.join(bo_PR_logs_folder, file)

            initial_indices = load_initial_point_indices(
                init_points_csv=init_points_csv,
                dataset=ds,
                feature_columns=ds.x_cols,
                n_initial_points=n_init_points,
            )

            gp_builder = GP(kernel_name="RBF", 
                random_state=seed, 
                length_scale=1.0)

            af = AF(
                kind="lcb",
                kappa=1.0,
                ml_on_bounds=False,
                ml_on_bounds_parameters={
                    "constraint_bounds": [[0, constr]],
                },
                ml_on_target=False,
                bounds=bounds,
                random_state=seed,
            )

            logger = Logger(dim=ds.dim)

            def dummy_objective(X):
                raise RuntimeError("Objective should not be called in dataset mode.")

            bo = BO(
                objective_function=dummy_objective,
                domain_bounds=bounds,
                gp_builder=gp_builder,
                af=af,
                constraint_functions=[],
                initial_points=3,
                init_strategy="random",
                random_state=seed,
                logger=logger,
                dataset=ds,
                epsilon_greedy=0.0,
                reparameterization=True
            )

            print(f"{path}/{seed}.csv")

            bo.initialize(X0=initial_indices)

            x_best, y_best = bo.run(n_iterations=30, n_restarts=10, verbose=True)
            print("Best feasible x =", x_best, " y =", y_best)

            bounds_metric = "mape"
            target_metric = "accuracy"
            bo.logger.to_csv(f"{path}/{seed}.csv", bounds_metric, target_metric)


    ### Stereomatch ###
    path = "output_with_initial_points_DiscreteBO/stereomatch"
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    n_init_points = 3

    filename = "resources/stereomatch.csv"
    x_cols = ["confidence","hypo_step","max_arm_length","num_threads"]

    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="cost",
        g_cols=["time"],
        t_col="time",
    )
    bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])

    constr_values = [8000, 10000, 12000, 17000, 20000, 40000]

    for constr in constr_values:

        path = f"output_with_initial_points_DiscreteBO/stereomatch/{constr}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        bo_PR_logs_folder = os.path.join(INPUT_FOLDER, 'stereomatch', str(constr))

        csv_files = [
                file for file in os.listdir(bo_PR_logs_folder)
                if file.lower().endswith(".csv")
            ]

        for file in csv_files:
            seed = int(file.split(".csv")[0])

            init_points_csv = os.path.join(bo_PR_logs_folder, file)

            initial_indices = load_initial_point_indices(
                init_points_csv=init_points_csv,
                dataset=ds,
                feature_columns=ds.x_cols,
                n_initial_points=n_init_points,
            )

            gp_builder = GP(kernel_name="RBF", 
                random_state=seed, 
                length_scale=1.0)

            af = AF(
                kind="lcb",
                kappa=1.0,
                ml_on_bounds=False,
                ml_on_bounds_parameters={
                    "constraint_bounds": [[0, constr]],
                },
                ml_on_target=False,
                bounds=bounds,
                random_state=seed,
            )

            logger = Logger(dim=ds.dim)

            def dummy_objective(X):
                raise RuntimeError("Objective should not be called in dataset mode.")

            bo = BO(
                objective_function=dummy_objective,
                domain_bounds=bounds,
                gp_builder=gp_builder,
                af=af,
                constraint_functions=[],
                initial_points=3,
                init_strategy="random",
                random_state=seed,
                logger=logger,
                dataset=ds,
                epsilon_greedy=0.0,
                reparameterization=True
            )

            print(f"{path}/{seed}.csv")

            bo.initialize(X0=initial_indices)

            x_best, y_best = bo.run(n_iterations=60, n_restarts=10, verbose=True)
            print("Best feasible x =", x_best, " y =", y_best)

            bounds_metric = "mape"
            target_metric = "accuracy"
            bo.logger.to_csv(f"{path}/{seed}.csv", bounds_metric, target_metric)


    ### LiGen ###
    path = "output_with_initial_points_DiscreteBO/ligen"
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    n_init_points = 11

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

    constr_values = [2.0, 2.1, 2.2, 2.45, 2.75]
    
    for constr in constr_values:

        path = f"output_with_initial_points_DiscreteBO/ligen/{constr}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        bo_PR_logs_folder = os.path.join(INPUT_FOLDER, 'ligen', str(constr))

        csv_files = [
                file for file in os.listdir(bo_PR_logs_folder)
                if file.lower().endswith(".csv")
            ]

        for file in csv_files:
            seed = int(file.split(".csv")[0])

            init_points_csv = os.path.join(bo_PR_logs_folder, file)

            initial_indices = load_initial_point_indices(
                init_points_csv=init_points_csv,
                dataset=ds,
                feature_columns=ds.x_cols,
                n_initial_points=n_init_points,
            )

            gp_builder = GP(kernel_name="RBF", 
                random_state=seed, 
                length_scale=1.0)

            af = AF(
                kind="lcb",
                kappa=1.0,
                ml_on_bounds=False,
                ml_on_bounds_parameters={
                    "constraint_bounds": [[0, constr]],
                },
                ml_on_target=False,
                bounds=bounds,
                random_state=seed,
            )

            logger = Logger(dim=ds.dim)

            def dummy_objective(X):
                raise RuntimeError("Objective should not be called in dataset mode.")

            bo = BO(
                objective_function=dummy_objective,
                domain_bounds=bounds,
                gp_builder=gp_builder,
                af=af,
                constraint_functions=[],
                initial_points=11,
                init_strategy="random",
                random_state=seed,
                logger=logger,
                dataset=ds,
                epsilon_greedy=0.0,
                reparameterization=True
            )

            print(f"{path}/{seed}.csv")

            bo.initialize(X0=initial_indices)

            x_best, y_best = bo.run(n_iterations=60, n_restarts=10, verbose=True)
            print("Best feasible x =", x_best, " y =", y_best)

            bounds_metric = "mape"
            target_metric = "accuracy"
            bo.logger.to_csv(f"{path}/{seed}.csv", bounds_metric, target_metric)


if __name__ == "__main__":
    main()