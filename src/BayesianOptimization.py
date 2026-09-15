import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from random import randint

from GaussianProcess import *
from AcquisitionFunction import *
from Logger import Logger
from Dataset import Dataset

class BO:
    def __init__(self, objective_function, domain_bounds, gp_builder, af, constraint_functions=None, initial_points=10,
                 random_state=16, logger=None, dataset=None, discrete_values=None, discrete_refine=True,
                 epsilon_greedy=0.0, epsilon_max_tries=200, epsilon_feas_threshold=0.5,
                 init_strategy="lhs", init_params=None, reparameterization=False,
                 lambda_max_variation=10.0, length_scale_bounds=(1e-9, 10.0), nu_bounds=(0.5, 10.0)):
        
        self.objective_function = objective_function
        self.domain_bounds = np.asarray(domain_bounds)
        self.dim = self.domain_bounds.shape[0]

        self.gp = gp_builder()
        self.gp_builder = gp_builder
        self.af = af
        self.constraint_functions = constraint_functions or []
        self.initial_points = initial_points
        
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.epsilon_greedy = float(epsilon_greedy)
        self.epsilon_max_tries = int(epsilon_max_tries)
        self.epsilon_feas_threshold = float(epsilon_feas_threshold)

        self.init_strategy = str(init_strategy).lower()
        valid_init_strategies = {"random", "lhs", "sobol"}
        if self.init_strategy not in valid_init_strategies:
            raise ValueError(
                f"Unknown init_strategy='{self.init_strategy}'. "
                f"Supported values are: {sorted(valid_init_strategies)}."
            )

        self.init_params = {} if init_params is None else dict(init_params)

        self.discrete_values = discrete_values
        self.discrete_refine = bool(discrete_refine)

        self.discrete_dims = []

        if self.discrete_values is not None:
            if len(self.discrete_values) != self.dim:
                raise ValueError("discrete_values must have length = dim")

            norm_vals = []
            for j, v in enumerate(self.discrete_values):
                if v is None:
                    norm_vals.append(None)
                    continue
                arr = np.sort(np.asarray(v, dtype=float).ravel())
                if arr.size == 0:
                    raise ValueError(f"discrete_values[{j}] is empty")
                norm_vals.append(arr)
                self.discrete_dims.append(j)

            self.discrete_values = norm_vals

        if hasattr(self.af, "rng"):
            self.af.rng = np.random.default_rng(random_state)
        if hasattr(self.gp, "random_state"):
            self.gp.random_state = random_state

        self.X_train = None
        self.y_train = None
        self.G_train = None  # (n, m) constraint values for X_train in non-dataset mode

        self.logger = logger if logger is not None else Logger(dim=self.dim)

        self.dataset = dataset
        if self.dataset is not None:
            self.dim = self.dataset.dim
            self.domain_bounds = np.column_stack([self.dataset.X.min(axis=0), self.dataset.X.max(axis=0)])
            self.used_idx = np.zeros(self.dataset.n, dtype=bool)
            self.train_idx = np.array([], dtype=int) 

        self._reparam_scale = np.where(
            self.domain_bounds[:, 1] > self.domain_bounds[:, 0],
            self.domain_bounds[:, 1] - self.domain_bounds[:, 0],
            1.0,
        )

        # Reparameterization parameters
        self.reparameterization = bool(reparameterization)
        self.lambda_max_variation = float(lambda_max_variation)
        self.length_scale_bounds = tuple(
            map(float, length_scale_bounds)
        )
        self.nu_bounds = tuple(map(float, nu_bounds))

        self.normalized_domain_diameter = np.sqrt(self.dim)
        self.reparameterization_penalty = 1.0 + np.sqrt(self.dim) + self.lambda_max_variation

    def _generate_initial_design(self, n, bounds=None, seed=None):
        """
        Draw n continuous points with random, LHS, or Sobol sampling.
        Points are not snapped to a discrete grid here.
        """
        n = int(n)

        if n < 0:
            raise ValueError("n must be non-negative.")

        if n == 0:
            return np.empty((0, self.dim), dtype=float)

        if bounds is None:
            bounds = self.domain_bounds

        bounds = np.asarray(bounds, dtype=float)

        if bounds.shape != (self.dim, 2):
            raise ValueError(
                f"bounds must have shape ({self.dim}, 2), got {bounds.shape}."
            )

        lower = bounds[:, 0]
        upper = bounds[:, 1]

        if np.any(upper < lower):
            raise ValueError(
                "Each upper bound must be greater than or equal to its lower bound."
            )

        if self.init_strategy == "random":
            return self.rng.uniform(
                lower,
                upper,
                size=(n, self.dim),
            )

        if seed is None:
            seed = self.random_state

        if self.init_strategy == "lhs":
            sampler = qmc.LatinHypercube(
                d=self.dim,
                seed=seed,
            )
            unit_design = sampler.random(n)

        elif self.init_strategy == "sobol":
            sampler = qmc.Sobol(
                d=self.dim,
                scramble=True,
                seed=seed,
            )
            unit_design = sampler.random(n)

        else:
            raise ValueError(
                f"Unknown init_strategy='{self.init_strategy}'. "
                "Supported strategies are: random, lhs, sobol."
            )

        return qmc.scale(
            unit_design,
            lower,
            upper,
        )
    
    def _snap_matrix_to_discrete(self, X):
        """
        Snap every row of X to the allowed values of discrete dimensions.
        Continuous dimensions are preserved unchanged.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f"X must have shape (n, {self.dim}), got {X.shape}."
            )

        if not self._has_discrete():
            return X.copy()

        return np.vstack([
            self._snap_to_discrete(row)
            for row in X
        ])

    def _fully_discrete_domain_size(self):
        """
        Return the number of configurations if every dimension is discrete.
        Return None if the domain has at least one continuous dimension.
        """
        if not self._has_discrete():
            return None

        if len(self.discrete_dims) != self.dim:
            return None

        return int(np.prod([
            self.discrete_values[j].size
            for j in self.discrete_dims
        ]))

    def _initial_points_non_dataset(self, n, existing_X=None):
        """
        Generate n unique valid configurations for non-dataset mode.

        Points are generated by random/LHS/Sobol, then snapped to the discrete
        grid where applicable. existing_X is excluded from the result.
        """
        n = int(n)

        if n < 0:
            raise ValueError("n must be non-negative.")

        if n == 0:
            return np.empty((0, self.dim), dtype=float)

        max_domain_size = self._fully_discrete_domain_size()

        existing_keys = set()

        if existing_X is not None:
            existing_X = np.asarray(existing_X, dtype=float)

            if existing_X.ndim != 2 or existing_X.shape[1] != self.dim:
                raise ValueError(
                    f"existing_X must have shape (m, {self.dim}), "
                    f"got {existing_X.shape}."
                )

            existing_X = self._snap_matrix_to_discrete(existing_X)

            for row in existing_X:
                existing_keys.add(
                    np.ascontiguousarray(row).tobytes()
                )

        if max_domain_size is not None:
            available_count = max_domain_size - len(existing_keys)

            if n > available_count:
                raise ValueError(
                    f"Requested {n} new initial points, but only "
                    f"{available_count} unvisited discrete configurations "
                    f"remain."
                )

        selected = []
        selected_keys = set(existing_keys)

        # A bounded retry count avoids infinite loops in pathological grids.
        max_rounds = int(
            self.init_params.get("max_init_rounds", 1_000)
        )

        for _ in range(max_rounds):
            if len(selected) >= n:
                break

            remaining = n - len(selected)
            n_draw = max(16, 2 * remaining)

            X_candidates = self._generate_initial_design(
                n_draw,
                bounds=self.domain_bounds,
            )

            X_candidates = self._snap_matrix_to_discrete(
                X_candidates
            )

            for row in X_candidates:
                key = np.ascontiguousarray(row).tobytes()

                if key in selected_keys:
                    continue

                selected.append(row.copy())
                selected_keys.add(key)

                if len(selected) == n:
                    break

        if len(selected) != n:
            raise RuntimeError(
                "Unable to generate the requested number of unique initial "
                "points. The discrete grid may be too small or excessively "
                "coarse after snapping."
            )

        return np.asarray(selected, dtype=float)

    def _initial_points_dataset(self, n, existing_idx=None):
        """
        Select n additional, unique dataset rows through random/LHS/Sobol.

        existing_idx is excluded and is not counted among the n requested rows.
        """
        if self.dataset is None:
            raise RuntimeError(
                "_initial_points_dataset requires dataset mode."
            )

        n = int(n)

        if n < 0:
            raise ValueError("n must be non-negative.")

        if existing_idx is None:
            existing_idx = np.empty(0, dtype=int)
        else:
            existing_idx = np.asarray(
                existing_idx,
                dtype=int,
            ).ravel()

        if np.unique(existing_idx).size != existing_idx.size:
            raise ValueError(
                "existing_idx contains duplicate dataset indices."
            )

        all_idx = np.arange(self.dataset.n, dtype=int)
        available_idx = np.setdiff1d(
            all_idx,
            existing_idx,
            assume_unique=False,
        )

        if n > available_idx.size:
            raise ValueError(
                f"Requested {n} initial rows, but only "
                f"{available_idx.size} unused dataset rows remain."
            )

        if n == 0:
            return np.empty(0, dtype=int)

        if self.init_strategy == "random":
            return self.rng.choice(
                available_idx,
                size=n,
                replace=False,
            ).astype(int)

        X_design = self._generate_initial_design(
            n=n,
            bounds=self.domain_bounds,
        )

        return self._snap_design_to_dataset_indices(
            X_design=X_design,
            available_idx=available_idx,
        )

    def _snap_design_to_dataset_indices(
        self,
        X_design,
        available_idx=None,
    ):
        """
        Map continuous design points to distinct valid dataset rows.

        Each design point is assigned to the closest currently available dataset
        configuration using normalized Euclidean distance. A selected row is
        removed from the available set, guaranteeing unique indices.

        Parameters
        ----------
        X_design : ndarray, shape (n, d)
            Design points produced by random, LHS, or Sobol sampling.

        available_idx : ndarray or None
            Dataset indices eligible for selection. If None, all rows are used.

        Returns
        -------
        ndarray, shape (n,)
            Selected unique dataset row indices.
        """
        if self.dataset is None:
            raise RuntimeError(
                "_snap_design_to_dataset_indices requires dataset mode."
            )

        X_design = np.asarray(X_design, dtype=float)

        if X_design.ndim != 2 or X_design.shape[1] != self.dim:
            raise ValueError(
                f"X_design must have shape (n, {self.dim}), "
                f"got {X_design.shape}."
            )

        if available_idx is None:
            available_idx = np.arange(self.dataset.n, dtype=int)
        else:
            available_idx = np.asarray(
                available_idx,
                dtype=int,
            ).ravel()

        if X_design.shape[0] > available_idx.size:
            raise ValueError(
                f"Cannot map {X_design.shape[0]} design points to only "
                f"{available_idx.size} available dataset rows."
            )

        X_data = self.dataset.X
        scale = self.domain_bounds[:, 1] - self.domain_bounds[:, 0]
        scale = np.where(scale > 0.0, scale, 1.0)

        available = available_idx.copy()
        selected_idx = []

        for x in X_design:
            X_available = X_data[available]

            distances = np.linalg.norm(
                (X_available - x) / scale,
                axis=1,
            )

            local_pos = int(np.argmin(distances))
            idx = int(available[local_pos])

            selected_idx.append(idx)
            available = np.delete(available, local_pos)

        return np.asarray(selected_idx, dtype=int)

    def sample_uniform(self, n):
        return self.rng.uniform(
            self.domain_bounds[:, 0],
            self.domain_bounds[:, 1],
            size=(n, self.dim),
        )

    def feasibility_mask(self, X):

        if not self.constraint_functions:
            return np.ones(X.shape[0], dtype=bool)

        mask = np.ones(X.shape[0], dtype=bool)
        for j, constr_fun in enumerate(self.constraint_functions):
            v = constr_fun(X).ravel()
            lb, ub = self.af.ml_on_bounds_parameters["constraint_bounds"][j]
            mask &= (v >= lb) & (v <= ub)
        return mask
    
    def feasibility_mask_idx(self, idx):
        idx = np.asarray(idx, dtype=int).ravel()
        if self.dataset is None or self.dataset.G is None:
            return np.ones(idx.shape[0], dtype=bool)

        G = self.dataset.G
        bounds = self.af.ml_on_bounds_parameters["constraint_bounds"]
        mask = np.ones(idx.shape[0], dtype=bool)
        for j in range(G.shape[1]):
            lb, ub = bounds[j]
            v = G[idx, j]
            mask &= (v >= lb) & (v <= ub)
        return mask

    def best_feasible_value(self):

        if self.dataset is not None:
            mask = self.feasibility_mask_idx(self.train_idx)
            if not np.any(mask):
                return None, mask
            feas_pos = np.where(mask)[0]
            best_pos = feas_pos[np.argmin(self.y_train[feas_pos])]
            return float(self.y_train[best_pos]), mask
    
        if self.G_train is not None:
            bounds = self.af.ml_on_bounds_parameters["constraint_bounds"]
            mask = np.ones(self.X_train.shape[0], dtype=bool)
            for j in range(self.G_train.shape[1]):
                lb, ub = bounds[j]
                v = self.G_train[:, j]
                mask &= (v >= lb) & (v <= ub)
        else:
            mask = self.feasibility_mask(self.X_train)

        if not np.any(mask):
            return None, mask
        feas_idx = np.where(mask)[0]
        best_idx = feas_idx[np.argmin(self.y_train[feas_idx])]
        return float(self.y_train[best_idx]), mask

    def fit_constraint_models(self):
        if not self.af.ml_on_bounds:
            return []

        task = self.af.ml_on_bounds_parameters["task"]
        constraint_metrics = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)

            if self.dataset is not None:
                G = self.dataset.G
                if G is None:
                    return []

                idx = self.train_idx

                for j in range(G.shape[1]):
                    v = G[idx, j]
                    lb, ub = self.af.ml_on_bounds_parameters["constraint_bounds"][j]
                    if task == "classification":
                        y_feas = ((v >= lb) & (v <= ub)).astype(int)
                        self.af.ml_bounds[j].fit(self.X_train, y_feas)
                        if np.unique(y_feas).size < 2:
                            metric = np.nan
                        else:
                            metric = self.af.ml_bounds[j].evaluate(self.X_train, y_feas)["accuracy"]
                    else:
                        self.af.ml_bounds[j].fit(self.X_train, v)
                        metric = self.af.ml_bounds[j].evaluate(self.X_train, v)["mape"]
                    constraint_metrics.append(metric)

                return constraint_metrics

            for j, constr_fun in enumerate(self.constraint_functions):
                
                if self.G_train is not None:
                    v = self.G_train[:, j].ravel()
                else:
                    v = constr_fun(self.X_train).ravel()

                lb, ub = self.af.ml_on_bounds_parameters["constraint_bounds"][j]

                if task == "classification":
                    y_feas = ((v >= lb) & (v <= ub)).astype(int)
                    self.af.ml_bounds[j].fit(self.X_train, y_feas)
                    if np.unique(y_feas).size < 2:
                        metric = np.nan
                    else:
                        metric = self.af.ml_bounds[j].evaluate(self.X_train, y_feas)["accuracy"]
                else:
                    self.af.ml_bounds[j].fit(self.X_train, v)
                    metric = self.af.ml_bounds[j].evaluate(self.X_train, v)["mape"]

                constraint_metrics.append(metric)

        return constraint_metrics

    def fit_target_model(self, y_best):
        if not self.af.ml_on_target:
            return {}

        task = self.af.ml_on_target_parameters["task"]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if task == "classification":
                y_feas = (self.y_train <= y_best).astype(int)
                self.af.ml_target.fit(self.X_train, y_feas)
                if np.unique(y_feas).size < 2:
                    target_metric = np.nan
                else:
                    target_metric = self.af.ml_target.evaluate(self.X_train, y_feas)["accuracy"]
            else:
                self.af.ml_target.fit(self.X_train, self.y_train)
                target_metric = self.af.ml_target.evaluate(self.X_train, self.y_train)["mape"]

        return target_metric
    
    def _ml_bounds_feasible(self, X):
        """
        Feasibility according to ML bounds models (if enabled and fitted).
        Returns boolean mask shape (n,).
        """
        X = np.asarray(X)
        if (not getattr(self.af, "ml_on_bounds", False)) or (not hasattr(self.af, "ml_bounds")):
            return np.ones(X.shape[0], dtype=bool)

        task = self.af.ml_on_bounds_parameters["task"]

        if task == "regression":
            mask = np.ones(X.shape[0], dtype=bool)
            for j, model in enumerate(self.af.ml_bounds):
                v = model.predict(X).ravel()
                lb, ub = self.af.ml_on_bounds_parameters["constraint_bounds"][j]
                mask &= (v >= lb) & (v <= ub)
            return mask

        # classification: require p(feasible=1) >= threshold for each constraint
        mask = np.ones(X.shape[0], dtype=bool)
        for j, model in enumerate(self.af.ml_bounds):
            p = model.predict_proba(X)[:, 1]
            mask &= (p >= self.epsilon_feas_threshold)
        return mask


    def _sample_random_point_continuous_or_mixed(self):
        """
        Random point in bounds, then enforce discrete dims if present.
        """
        x = self.rng.uniform(self.domain_bounds[:, 0], self.domain_bounds[:, 1]).astype(float)
        if self._has_discrete():
            for j in self.discrete_dims:
                x[j] = float(self.rng.choice(self.discrete_values[j]))
        return x.reshape(1, -1)


    def _epsilon_pick_continuous_or_mixed(self, y_best):
        """
        Try to sample a random point predicted feasible by ML bounds.
        Returns x_next (1,d) or None if not found.
        """
        for _ in range(self.epsilon_max_tries):
            x = self._sample_random_point_continuous_or_mixed()

            # keep your discrete pipeline consistent
            if self._has_discrete():
                x = self._snap_to_discrete(x)
                if self.discrete_refine:
                    x = self._refine_discrete_by_af(x, y_best=y_best)

            if self._is_duplicate(x, tol=0.0):
                continue

            if self._ml_bounds_feasible(x)[0]:
                return x

        return None


    def _epsilon_pick_dataset_index(self, idx_pool):
        """
        idx_pool: 1D array of available candidate indices (unused rows)
        Returns idx_next or None.
        """
        idx_pool = np.asarray(idx_pool, dtype=int).ravel()
        if idx_pool.size == 0:
            return None

        X_pool = self.dataset.X[idx_pool]

        # if ML bounds not enabled, this degenerates to uniform random unused row
        feas = self._ml_bounds_feasible(X_pool)

        if np.any(feas):
            return int(self.rng.choice(idx_pool[feas]))

        # fallback: no predicted-feasible rows -> random unused row
        return int(self.rng.choice(idx_pool))
    

    def compute_global_best_feasible_from_dataset(self):
        if self.dataset is None or self.dataset.G is None:
            self.global_best_feasible_idx = None
            self.global_best_feasible_x = None
            self.global_best_feasible_y = None
            return

        G = self.dataset.G
        y = self.dataset.y

        if self.af.ml_on_bounds:
            bounds = self.af.ml_on_bounds_parameters["constraint_bounds"]
        else:
            bounds = [(float("-inf"), float("inf"))] * G.shape[1]

        mask = np.ones(self.dataset.n, dtype=bool)
        for j in range(G.shape[1]):
            lb, ub = bounds[j]
            mask &= (G[:, j] >= lb) & (G[:, j] <= ub)

        if not np.any(mask):
            self.global_best_feasible_idx = None
            self.global_best_feasible_x = None
            self.global_best_feasible_y = None
            return

        feas_idx = np.where(mask)[0]
        best_idx = feas_idx[np.argmin(y[feas_idx])]

        self.global_best_feasible_idx = int(best_idx)
        self.global_best_feasible_x = self.dataset.X[best_idx].copy()
        self.global_best_feasible_y = float(y[best_idx])


    def initialize(self, X0=None, y0=None, G0=None):
        """
        Initialize BO observations.

        Non-dataset mode
        ----------------
        - Generates points using init_strategy in {"random", "lhs", "sobol"}.
        - Snaps dimensions declared in discrete_values to their permitted values.
        - Enforces uniqueness after snapping.
        - X0 is an optional arbitrary warm start with shape (n, dim).

        Dataset mode
        ------------
        - Generates a design using init_strategy over the dataset bounding box.
        - For LHS/Sobol, maps each design point to a distinct closest row of
        dataset.X using normalized Euclidean distance.
        - For random, samples unused dataset rows without replacement.
        - X0 may be either dataset row indices, shape (n,), or exact rows from
        dataset.X, shape (n, dim).
        - y0 and G0 are ignored because y/G are already stored in the dataset.
        """
        if self.dataset is not None:
            X_cand = self.dataset.X
            y_cand = self.dataset.y
            n_dataset = self.dataset.n

            self.compute_global_best_feasible_from_dataset()

            if X0 is None:
                n_initial = min(int(self.initial_points), n_dataset)

                idx0 = self._initial_points_dataset(
                    n=n_initial,
                )

            else:
                if y0 is not None or G0 is not None:
                    warnings.warn(
                        "y0 and G0 are ignored in dataset mode because the "
                        "objective and constraints are read from the dataset.",
                        UserWarning,
                    )

                idx_user = self._dataset_points_to_indices(X0)

                if np.unique(idx_user).size != idx_user.size:
                    raise ValueError(
                        "Duplicate dataset indices/configurations in X0."
                    )

                n_initial = min(
                    max(int(self.initial_points), idx_user.size),
                    n_dataset,
                )

                n_extra = n_initial - idx_user.size

                idx_extra = self._initial_points_dataset(
                    n=n_extra,
                    existing_idx=idx_user,
                )

                idx0 = np.concatenate([
                    idx_user,
                    idx_extra,
                ])

            self.used_idx[idx0] = True
            self.train_idx = np.asarray(idx0, dtype=int)

            self.X_train = np.asarray(
                X_cand[idx0],
                dtype=float,
            )

            self.y_train = np.asarray(
                y_cand[idx0],
                dtype=float,
            ).ravel()

            self.G_train = (
                np.asarray(
                    self.dataset.G[idx0],
                    dtype=float,
                )
                if self.dataset.G is not None
                else None
            )

            if self.logger is not None:
                feasible = self.feasibility_mask_idx(idx0)
                y_best_feasible, _ = self.best_feasible_value()

                for i in range(self.X_train.shape[0]):
                    self.logger.log(
                        iter=0,
                        x_next=self.X_train[i],
                        y_next=self.y_train[i],
                        feasible=bool(feasible[i]),
                        y_best_feasible=y_best_feasible,
                    )

            return

        # ---------------------------------------------------------------
        # Non-dataset mode
        # ---------------------------------------------------------------
        if X0 is not None:
            X0, y0, G0 = self._check_warmstart_shapes(
                X0,
                y0=y0,
                G0=G0,
            )

            # Snap custom warm-start points before checking uniqueness.
            X0 = self._snap_matrix_to_discrete(X0)

            warm_start_keys = [
                np.ascontiguousarray(
                    row,
                    dtype=float,
                ).tobytes()
                for row in X0
            ]

            if len(set(warm_start_keys)) != X0.shape[0]:
                raise ValueError(
                    "Duplicate points in X0 after discrete snapping."
                )

            n_initial = max(
                int(self.initial_points),
                X0.shape[0],
            )

            X_extra = self._initial_points_non_dataset(
                n=n_initial - X0.shape[0],
                existing_X=X0,
            )

            self.X_train = np.vstack([
                X0,
                X_extra,
            ])

            # If y0 is unavailable, evaluate every initial point.
            if y0 is None:
                self.y_train = np.asarray(
                    self.objective_function(self.X_train),
                    dtype=float,
                ).ravel()

            # If y0 is supplied, evaluate only the points generated by BO.
            else:
                y0 = np.asarray(
                    y0,
                    dtype=float,
                ).ravel()

                y_extra = (
                    np.asarray(
                        self.objective_function(X_extra),
                        dtype=float,
                    ).ravel()
                    if X_extra.shape[0] > 0
                    else np.empty(0, dtype=float)
                )

                self.y_train = np.concatenate([
                    y0,
                    y_extra,
                ])

            n_constraints = self._n_constraints()

            if n_constraints == 0:
                self.G_train = None

            elif G0 is None:
                self.G_train = np.asarray(
                    self._compute_constraints(self.X_train),
                    dtype=float,
                )

            else:
                G0 = np.asarray(
                    G0,
                    dtype=float,
                )

                G_extra = (
                    np.asarray(
                        self._compute_constraints(X_extra),
                        dtype=float,
                    )
                    if X_extra.shape[0] > 0
                    else np.empty(
                        (0, n_constraints),
                        dtype=float,
                    )
                )

                self.G_train = np.vstack([
                    G0,
                    G_extra,
                ])

        # No warm start: generate all initial points.
        else:
            self.X_train = self._initial_points_non_dataset(
                n=int(self.initial_points),
            )

            self.y_train = np.asarray(
                self.objective_function(self.X_train),
                dtype=float,
            ).ravel()

            self.G_train = self._compute_constraints(
                self.X_train
            )

        if self.logger is not None:
            y_best_feasible, feasible = self.best_feasible_value()

            for i in range(self.X_train.shape[0]):
                self.logger.log(
                    iter=0,
                    x_next=self.X_train[i],
                    y_next=self.y_train[i],
                    feasible=bool(feasible[i]),
                    y_best_feasible=y_best_feasible,
                )

    def _dataset_points_to_indices(self, X0):
        """
        Convert dataset indices or exact dataset rows to indices.

        This intentionally requires exact row matches if X0 is supplied as
        configurations, because approximate matching can silently select an
        unintended experiment.
        """
        X0 = np.asarray(X0)

        if X0.ndim == 1:
            idx = X0.astype(int).ravel()

            if np.any(idx < 0) or np.any(idx >= self.dataset.n):
                raise IndexError(
                    "At least one X0 dataset index is out of range."
                )

            return idx

        if X0.ndim != 2 or X0.shape[1] != self.dim:
            raise ValueError(
                f"In dataset mode, X0 must be a 1D array of indices "
                f"or an array with shape (n, {self.dim})."
            )

        dataset_rows = {
            np.ascontiguousarray(
                row,
                dtype=float,
            ).tobytes(): idx
            for idx, row in enumerate(self.dataset.X)
        }

        indices = []

        for row in np.asarray(X0, dtype=float):
            key = np.ascontiguousarray(row).tobytes()

            if key not in dataset_rows:
                raise ValueError(
                    "A row in X0 does not exactly match a row in dataset.X. "
                    "Pass dataset indices instead, or use an exact dataset row."
                )

            indices.append(dataset_rows[key])

        return np.asarray(indices, dtype=int)

    def step(self, n_restarts=10, iter_idx=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.X_train, self.y_train)

        best_feas, _ = self.best_feasible_value()
        y_best = best_feas if best_feas is not None else float(np.inf)

        constraint_metrics = self.fit_constraint_models()
        target_metrics = self.fit_target_model(y_best)

        if self.dataset is not None:
            X_cand = self.dataset.X
            y_cand = self.dataset.y

            pool_mask = ~self.used_idx
            if not np.any(pool_mask):
                raise RuntimeError("No new candidate rows left in the dataset.")

            idx_pool = np.where(pool_mask)[0]

            if self.epsilon_greedy > 0.0 and (self.rng.random() < self.epsilon_greedy):
                idx_next = self._epsilon_pick_dataset_index(idx_pool)
            else:
                if not self.reparameterization:
                    X_pool = X_cand[idx_pool]

                    pos = self.af.maximise_over_dataset(
                        gp=self.gp,
                        X_cand=X_pool,
                        y_best=y_best,
                        return_index=True,
                    )

                    idx_next = int(idx_pool[int(pos)])

                else:
                    pos = self.af.maximise_over_dataset(
                        gp=self.gp,
                        X_cand=X_cand,
                        y_best=y_best,
                        return_index=True,
                        used_idx=np.where(self.used_idx)[0]
                    )

                    idx_cand = int(pos)

                    if self.used_idx[idx_cand]:
                        idx_next = self._reparameterization_strategy(
                            idx_cand=idx_cand,
                            y_best=y_best,
                        )
                    else:
                        idx_next = idx_cand

            self.used_idx[idx_next] = True
            self.train_idx = np.append(self.train_idx, idx_next)

            x_next = X_cand[idx_next].reshape(1, -1)
            y_next = float(y_cand[idx_next])

            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            if self.dataset.G is not None:
                g_next = np.asarray(self.dataset.G[idx_next], dtype=float).reshape(1, -1)
                self.G_train = g_next if self.G_train is None else np.vstack([self.G_train, g_next])

            is_feasible = bool(self.feasibility_mask_idx([idx_next])[0])

        else:
            if self.epsilon_greedy > 0.0 and (self.rng.random() < self.epsilon_greedy):
                x_eps = self._epsilon_pick_continuous_or_mixed(y_best=y_best)
                if x_eps is not None:
                    x_next = x_eps
                else:
                    x_next = None
            else:
                x_next = None

            if x_next is None:
                x_next_cont = self.af.maximise(
                    gp=self.gp,
                    bounds=self.domain_bounds,
                    y_best=y_best,
                    n_restarts=n_restarts
                ).reshape(1, -1)

                if self._has_discrete():
                    x_next = self._snap_to_discrete(x_next_cont)
                    if self.discrete_refine:
                        x_next = self._refine_discrete_by_af(x_next, y_best=y_best)

                    if self._is_duplicate(x_next, tol=0.0):
                        #import pdb; pdb.set_trace()     # TO FIX
                        x_next = self._random_mixed_point(x_base=x_next_cont)
                else:
                    x_next = x_next_cont

            y_next = float(self.objective_function(x_next).ravel()[0])

            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)

            if self.dataset is None and self._n_constraints() > 0:
                g_next = np.array([cf(x_next).ravel()[0] for cf in self.constraint_functions], dtype=float).reshape(1, -1)
                if self.G_train is None:
                    self.G_train = g_next
                else:
                    self.G_train = np.vstack([self.G_train, g_next])

            mask_next = self.feasibility_mask(x_next)
            is_feasible = bool(mask_next[0])

        best_feas_after, _ = self.best_feasible_value()

        if self.logger is not None:
            self.logger.log(
                iter=iter_idx if iter_idx is not None else len(self.logger),
                x_next=x_next,
                y_next=y_next,
                feasible=is_feasible,
                y_best_feasible=best_feas_after,
                accuracy_ml_bounds=constraint_metrics,
                accuracy_ml_target=target_metrics,
            )

        return x_next, y_next, y_best

    def _reparameterization_strategy(self, idx_cand, y_best):

        if self.dataset is None:
            raise RuntimeError(
                "Reparameterization requires dataset mode."
            )

        print("\n Starting reparameterization strategy... \n")

        kernel_name = self.gp_builder.kernel_name.lower()
        af_kind = self.af.kind.lower()

        lambda_current = self._current_lambda()
        has_af_lambda = (af_kind == "lcb" and lambda_current is not None)

        if kernel_name == "rbf":
            k_lower, k_upper = self.length_scale_bounds
        elif kernel_name == "matern":
            k_lower, k_upper = self.nu_bounds
        else:
            raise ValueError(
                f"Unsupported kernel: {kernel_name}"
            )

        #k0 = self._current_kernel_parameter()
        x_original = self.dataset.X[idx_cand]

        if has_af_lambda:
            #theta0 = np.array([k0, lambda_current], dtype=float)
            bounds = [(k_lower, k_upper), (lambda_current, lambda_current + self.lambda_max_variation)]

        else:
            #theta0 = np.array([k0], dtype=float)
            bounds = [(k_lower, k_upper)]

        self._best_reparameterization_trial = None

        def objective(theta):
            return self._reparameterization_objective(
                theta=theta,
                x_original=x_original,
                y_best=y_best,
                lambda_current=lambda_current,
                has_af_lambda=has_af_lambda,
            )

        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=self.random_state,
            strategy="best1bin",
            maxiter=30,
            popsize=8,
            polish=False,
            workers=1,
        )

        best_trial = self._best_reparameterization_trial

        if best_trial["is_visited"]:
            warnings.warn(
                "Reparameterization search failed to locate an unvisited AF "
                "Falling back to direct AF ranking over unvisited candidates.",
                RuntimeWarning,
            )
            idx_next = self._fallback_best_unvisited(y_best=y_best)
            return idx_next

        theta_opt = best_trial["theta"]
        idx_next = int(best_trial["idx"])

        kernel_param_opt = float(theta_opt[0])

        if has_af_lambda:
            lambda_param_opt = float(theta_opt[1])
        else:
            lambda_param_opt = None

        # Build the reparameterized GP and AF
        if kernel_name == "rbf":
            gp_opt = self.gp_builder.clone_GP(
                length_scale=kernel_param_opt,
                optimize_hyperparameters=False,
            )
        elif kernel_name == "matern":
            gp_opt = self.gp_builder.clone_GP(
                nu=kernel_param_opt,
                optimize_hyperparameters=False,
            )
        else:
            raise ValueError(
                f"Unsupported kernel: {kernel_name}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp_opt.fit(self.X_train, self.y_train)

        if af_kind == "ei":
            af_opt = self.af.clone_AF()
        elif af_kind == "lcb":
            af_opt = self.af.clone_AF(
                kappa=lambda_param_opt
            )
        else:
            raise ValueError(
                f"Unsupported AF: {af_kind}"
            )

        self.gp = gp_opt
        self.af = af_opt

        print(self.gp, self.af.af_kwards.get("kappa", 1.0))

        return idx_next

    def _fallback_best_unvisited(self, y_best):

        print("\n Fallback from DE... \n")

        if np.all(self.used_idx):
            raise RuntimeError(
                "No unvisited candidates remain; cannot fall back."
            )

        unvisited_idx = np.where(~self.used_idx)[0]

        idx_local = int(
            self.af.maximise_over_dataset(
                gp=self.gp,
                X_cand=self.dataset.X[unvisited_idx],
                y_best=y_best,
                return_index=True,
            )
        )

        idx_next = int(unvisited_idx[idx_local])

        return idx_next
    
    def _reparameterization_objective(self, theta, x_original, y_best, lambda_current, has_af_lambda):

        kernel_name = self.gp_builder.kernel_name.lower()
        af_kind = self.af.kind.lower()

        kernel_param = float(theta[0])
        lambda_param = (float(theta[1]) if has_af_lambda else None)

        if kernel_name == "rbf":
            gp_trial = self.gp_builder.clone_GP(
                length_scale=kernel_param,
                optimize_hyperparameters=False,
            )
        elif kernel_name == "matern":
            gp_trial = self.gp_builder.clone_GP(
                nu=kernel_param,
                optimize_hyperparameters=False,
            )
        else:
            raise ValueError(
                f"Unsupported kernel: {kernel_name}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                ConvergenceWarning,
            )
            gp_trial.fit(self.X_train, self.y_train)

        if af_kind == "ei":
            af_trial = self.af.clone_AF()
            parameter_cost = 0.0
        elif af_kind == "lcb":
            af_trial = self.af.clone_AF(kappa=lambda_param)
            parameter_cost = abs(lambda_param - lambda_current)
        else:
            raise ValueError(
                f"Unsupported AF: {af_kind}"
            )

        idx_trial = int(
            af_trial.maximise_over_dataset(
                gp=gp_trial,
                X_cand=self.dataset.X,
                y_best=y_best,
                return_index=True,
                used_idx=np.where(self.used_idx)[0]
            )
        )

        x_trial = self.dataset.X[idx_trial]
        is_visited = bool(self.used_idx[idx_trial])

        distance_cost = float(np.linalg.norm((x_original - x_trial) / self._reparam_scale))
        objective_value = (parameter_cost + distance_cost + (self.reparameterization_penalty if is_visited else 0.0))

        current_best = self._best_reparameterization_trial

        if (current_best is None or objective_value < current_best["value"]):
            self._best_reparameterization_trial = {
                "value": float(objective_value),
                "theta": np.asarray(theta, dtype=float).copy(),
                "idx": int(idx_trial),
                "is_visited": bool(is_visited),
            }

        return float(objective_value)

    def _current_kernel_parameter(self):
        kernel = self.gp.kernel_

        if self.gp_builder.kernel_name.lower() == "rbf":
            value = np.asarray(
                kernel.length_scale,
                dtype=float,
            ).ravel()

        elif self.gp_builder.kernel_name.lower() == "matern":
            value = np.asarray(
                kernel.nu,
                dtype=float,
            ).ravel()

        else:
            raise ValueError(
                "Unsupported kernel."
            )

        return float(value[0])

    def _current_lambda(self):
        if self.af.kind.lower() == "ei":
            return None

        return float(
            self.af.af_kwards.get("kappa", 1.0)
        )

    def run(self, n_iterations=20, n_restarts=10, verbose=True):

        for i in range(n_iterations):
            x_next, y_next, _ = self.step(n_restarts=n_restarts, iter_idx=i+1)

            if verbose:
                best_feas, _ = self.best_feasible_value()
                print(
                    f"Iter {i+1:02d} | "
                    f"x_next = {x_next.ravel()} | "
                    f"y_next = {y_next:.4f} | "
                    f"best_feasible = {best_feas if best_feas is not None else np.nan}"
                )

        best_feas, mask = self.best_feasible_value()
        if best_feas is not None:
            feas_idx = np.where(mask)[0]
            best_idx = feas_idx[np.argmin(self.y_train[feas_idx])]
        else:
            raise ValueError("No feasible point found.")

        self.best_idx = best_idx

        return self.X_train[best_idx], self.y_train[best_idx]
    
    def _has_discrete(self):
        return (self.discrete_values is not None) and (len(self.discrete_dims) > 0)

    def _snap_to_discrete(self, x):
        x = np.asarray(x).reshape(-1).astype(float, copy=False)
        if not self._has_discrete():
            return x.reshape(1, -1)

        snapped = x.copy()
        for j in self.discrete_dims:
            vals = self.discrete_values[j]
            k = int(np.argmin(np.abs(vals - snapped[j])))
            snapped[j] = vals[k]
        return snapped.reshape(1, -1)

    def _is_duplicate(self, x, tol=0.0):
        if self.X_train is None or len(self.X_train) == 0:
            return False
        x = np.asarray(x).reshape(1, -1)
        if tol == 0.0:
            return bool(np.any(np.all(self.X_train == x, axis=1)))
        return bool(np.any(np.all(np.isclose(self.X_train, x, atol=tol, rtol=0.0), axis=1)))

    def _random_mixed_point(self, x_base=None):
        """
        Randomize only discrete dimensions; keep continuous ones near x_base.
        If x_base is None, sample continuous dims uniformly in bounds.
        """
        if not self._has_discrete():
            raise RuntimeError("No discrete dimensions are defined.")

        if x_base is None:
            x = self.rng.uniform(self.domain_bounds[:, 0], self.domain_bounds[:, 1]).astype(float)
        else:
            x = np.asarray(x_base).reshape(-1).astype(float, copy=True)

        for j in self.discrete_dims:
            x[j] = float(self.rng.choice(self.discrete_values[j]))

        return x.reshape(1, -1)

    def _refine_discrete_by_af(self, x0, y_best):
        """
        Local discrete search (2*|D|+1 points): move each discrete dim by +/- one level.
        Works for mixed variables because it touches only discrete coords.
        """
        x0 = np.asarray(x0).reshape(1, -1)

        if not self._has_discrete():
            return x0

        cand = [x0.copy()]
        for j in self.discrete_dims:
            vals = self.discrete_values[j]
            k = int(np.argmin(np.abs(vals - x0[0, j])))

            if k - 1 >= 0:
                xm = x0.copy()
                xm[0, j] = vals[k - 1]
                cand.append(xm)

            if k + 1 < vals.size:
                xp = x0.copy()
                xp[0, j] = vals[k + 1]
                cand.append(xp)

        Xcand = np.vstack(cand)

        if self.X_train is not None and len(self.X_train) > 0:
            dup = np.array([self._is_duplicate(Xcand[i], tol=0.0) for i in range(Xcand.shape[0])], dtype=bool)
            if not np.all(dup):
                Xcand = Xcand[~dup]

        if Xcand.shape[0] == 0:
            return x0

        af_vals = self.af(Xcand, gp=self.gp, y_best=y_best).ravel()
        jbest = int(np.argmax(af_vals))
        return Xcand[jbest].reshape(1, -1)
    
    def _n_constraints(self):
        if self.dataset is not None and self.dataset.G is not None:
            return int(self.dataset.G.shape[1])
        return int(len(self.constraint_functions))

    def _compute_constraints(self, X):
        X = np.asarray(X)
        m = self._n_constraints()
        if m == 0:
            return None
        G = np.column_stack([cf(X).ravel() for cf in self.constraint_functions])
        return np.asarray(G, dtype=float)

    def _check_warmstart_shapes(self, X0, y0=None, G0=None):
        X0 = np.asarray(X0, dtype=float)
        if X0.ndim != 2 or X0.shape[1] != self.dim:
            raise ValueError(f"X0 must have shape (n,{self.dim}), got {X0.shape}")
        n0 = X0.shape[0]

        if y0 is not None:
            y0 = np.asarray(y0, dtype=float).ravel()
            if y0.shape[0] != n0:
                raise ValueError(f"y0 must have length {n0}, got {y0.shape[0]}")

        if G0 is not None:
            G0 = np.asarray(G0, dtype=float)
            if G0.ndim == 1:
                G0 = G0.reshape(-1, 1)
            m = self._n_constraints()
            if G0.shape != (n0, m):
                raise ValueError(f"G0 must have shape ({n0},{m}), got {G0.shape}")

        return X0, y0, G0



def main():
    
    def objective(X):
        x1, x2 = X[:, 0], X[:, 1]
        return np.sin(x1) + np.cos(x2) + 0.1 * (x1 - 2) ** 2


    bounds = np.array([
        [0.0, 5.0],
        [0.0, 5.0],
    ])

    def g1(X): return X[:, 0] + X[:, 1]
    def g2(X): return X[:, 0] * X[:, 1]


    def run_case(case_name, discrete_values=None, random_state=16):
        print(f"\n=== {case_name} ===")

        gp_builder = GP(kernel_name="Matern", 
                        random_state=random_state, 
                        length_scale=1.0)

        af = AF(
            kind="ei",
            kappa=1.0,
            ml_on_bounds=True,
            ml_on_bounds_parameters={
                "name": "ridge",
                "task": "regression",
                "constraint_bounds": [[1, 4], [1, 4]],
            },
            ml_on_target=True,
            ml_on_target_parameters={
                "name": "ridge",
                "task": "classification",
            },
            bounds=bounds,
            random_state=random_state,
        )

        logger = Logger(bounds.shape[0])

        bo = BO(
            objective_function=objective,
            domain_bounds=bounds,
            gp_builder=gp_builder,
            af=af,
            constraint_functions=[g1, g2],
            initial_points=10,
            logger=logger,
            random_state=random_state,
            discrete_values=discrete_values,
            discrete_refine=True
        )

        bo.initialize()
        x_best, y_best = bo.run(n_iterations=20, n_restarts=10, verbose=True)
        print("Best feasible: x =", x_best, ", y =", y_best)

        if af.ml_on_bounds:
            bounds_metric = "mape" if af.ml_on_bounds_parameters["task"] == "regression" else "accuracy"
        else:
            bounds_metric = ""

        if af.ml_on_target:
            target_metric = "mape" if af.ml_on_target_parameters["task"] == "regression" else "accuracy"
        else:
            target_metric = ""

        bo.logger.to_csv(f"test_{case_name}.csv", bounds_metric, target_metric)

    # run_case(
    #     case_name="continuous",
    #     discrete_values=None,              # or [None, None]
    #     random_state=randint(1,1000)
    # )
    
    # run_case(
    #     case_name="mixed_x2_discrete",
    #     discrete_values=[
    #         None,                          # x1 continuous
    #         np.linspace(0.0, 5.0, 11),     # x2 in {0.0, 0.5, 1.0, ..., 5.0}
    #     ],
    #     random_state=randint(1,1000)
    # )

    # run_case(
    #     case_name="discrete_both",
    #     discrete_values=[
    #         np.linspace(0.0, 5.0, 11),     # x1 in {0.0, 0.5, ..., 5.0}
    #         np.linspace(0.0, 5.0, 11),     # x2 in {0.0, 0.5, ..., 5.0}
    #     ],
    #     random_state=randint(1,1000)
    # )

    print(f"\n=== ligen ===")
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

    # print(f"\n=== oscarp ===")
    # filename = "resources/oscarp.csv"

    # x_cols = [
    #     "parallelism_ffmpeg-0",
    #     "parallelism_librosa",
    #     "parallelism_ffmpeg-1",
    #     "parallelism_ffmpeg-2",
    #     "parallelism_deepspeech"
    # ]

    # ds = Dataset.from_file(
    #     filename,
    #     x_cols=x_cols,
    #     y_col="cost",
    #     g_cols=["total_time"],
    #     t_col="total_time",
    # )

    bounds = np.column_stack([ds.X.min(axis=0), ds.X.max(axis=0)])
    random_state = randint(1,1000)

    gp_builder = GP(kernel_name="RBF", 
                    random_state=random_state, 
                    length_scale=1.0)

    af = AF(
        kind="lcb",
        kappa=1.0,
        ml_on_bounds=True,
        ml_on_bounds_parameters={
            "name": "ridge",
            "task": "regression",
            "constraint_bounds": [(0.0, 2.0)],
        },
        ml_on_target=True,
        ml_on_target_parameters={
            "name": "ridge",
            "task": "classification",
            #"hidden_layer_sizes": (100, 50),
            #"max_iter": 500
        },
        bounds=bounds,
        random_state=random_state,
    )

    logger = Logger(dim=ds.dim)

    def dummy_objective(X):
        raise RuntimeError("Objective should not be called in dataset mode.")

    bo = BO(
        objective_function=dummy_objective,
        domain_bounds=bounds,
        gp_builder=gp_builder,
        af=af,
        constraint_functions=[],     # constraints come from ds.G in dataset mode
        initial_points=10,
        init_strategy="lhs",
        random_state=random_state,
        logger=logger,
        dataset=ds,
        epsilon_greedy=0.1,
        reparameterization=True
    )

    X_init = np.array([
        [ 8,  8, 1, 128, 1024, 256, 1, 20],
        [32, 12, 2, 192,  256,  10, 4, 10],
        [72, 12, 1, 256,  256,  30, 4, 50],
        [72, 48, 3, 224,  256,  30, 2,  2],
        [20, 48, 1,  64,  256,  10, 1, 20],
        [16, 32, 2, 128, 1024,  10, 3,  1],
        [12,  8, 2, 160,  256,  30, 4,  2],
        [16, 24, 1, 128, 1024, 256, 1, 20],
        [32, 20, 1, 224,  256,  50, 1,  2],
        [20, 72, 3, 256, 1024,  30, 3, 20],
    ], dtype=float)

    bo.initialize()
    #bo.initialize(X0=X_init)    # samples initial points from ds.X / ds.y

    x_best, y_best = bo.run(n_iterations=200, n_restarts=10, verbose=True)
    print("Best feasible x =", x_best, " y =", y_best)

    # CSV metrics naming
    bounds_metric = "accuracy"   # regression constraint models
    target_metric = "mape"       # no mlontarget here
    bo.logger.to_csv("ligen_log.csv", bounds_metric, target_metric)
    

if __name__ == "__main__":
    main()
