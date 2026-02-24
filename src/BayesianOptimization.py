import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from GaussianProcess import *
from AcquisitionFunction import *
from Logger import Logger
from Dataset import Dataset

class BO:
    def __init__(self, objective_function, domain_bounds, gp, af, constraint_functions=None, initial_points=10,
                 random_state=16, logger=None, dataset=None, discrete_values=None, discrete_refine=True,
                 epsilon_greedy=0.0, epsilon_max_tries=200, epsilon_feas_threshold=0.5,
                 init_strategy="random", init_params=None):
        
        self.objective_function = objective_function
        self.domain_bounds = np.asarray(domain_bounds)
        self.dim = self.domain_bounds.shape[0]

        self.gp = gp
        self.af = af
        self.constraint_functions = constraint_functions or []
        self.initial_points = initial_points
        
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.epsilon_greedy = float(epsilon_greedy)
        self.epsilon_max_tries = int(epsilon_max_tries)
        self.epsilon_feas_threshold = float(epsilon_feas_threshold)

        self.init_strategy = str(init_strategy).lower()
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
        mask = np.ones(idx.shape[0], dtype=bool)
        for j in range(G.shape[1]):
            lb, ub = self.af.ml_on_bounds_parameters["constraint_bounds"][j]
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
        bounds = self.af.ml_on_bounds_parameters["constraint_bounds"]

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

    def _initial_design_continuous(self, n, strategy):
        bounds = np.asarray(self.domain_bounds, dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = self.dim

        if strategy == "random":
            return self.sample_uniform(n)

        from scipy.stats import qmc

        seed = int(self.random_state)
        if strategy == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            U = sampler.random(n)
        elif strategy == "sobol":
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            U = sampler.random(n)
        else:
            raise ValueError(f"Unknown init_strategy='{strategy}'")

        return qmc.scale(U, lo, hi)

    def initialize(self, X0=None, y0=None, G0=None):
        """
        Initialization with three modes:

        1) Dataset mode (self.dataset is not None):
        - If X0 is None: behaves like before (random subset without replacement).
        - If X0 is provided: X0 must be the PREFIX of what initialize(None) would have selected,
            so that the extra points are exactly identical to initialize(None). X0 is expected as indices.

        2) Non-dataset mode, warm-start (X0 is not None):
        - Enforce self.initial_points >= X0.shape[0] (if not, increase it).
        - Extra points are EXACTLY the tail of initialize(None) for that n_target.
        - No duplicates (uniqueness enforced after discrete snapping).

        3) Non-dataset mode, no warm-start (X0 is None):
        - Generates self.initial_points unique points (after discrete snapping).
        """

        def _key_row(x_row):
            x_row = np.asarray(x_row, dtype=float).reshape(-1)
            return x_row.tobytes()

        def _maybe_snap(X):
            X = np.asarray(X, dtype=float)
            if not self._has_discrete():
                return X
            return np.vstack([self._snap_to_discrete(X[i]) for i in range(X.shape[0])])

        def _generate_full_initial_design(n_total):
            n_total = int(n_total)
            if n_total <= 0:
                return np.zeros((0, self.dim), dtype=float)

            seen = set()
            X_out = []

            def _draw_candidates(n_draw):
                n_draw = int(n_draw)
                if n_draw <= 0:
                    return np.zeros((0, self.dim), dtype=float)

                if self.init_strategy == "random":
                    return self.sample_uniform(n_draw)

                from scipy.stats import qmc
                bounds = np.asarray(self.domain_bounds, dtype=float)
                lo, hi = bounds[:, 0], bounds[:, 1]
                d = self.dim
                seed = int(self.random_state)

                if self.init_strategy == "lhs":
                    if not hasattr(_draw_candidates, "_sampler"):
                        _draw_candidates._sampler = qmc.LatinHypercube(d=d, seed=seed)
                    U = _draw_candidates._sampler.random(n_draw)

                elif self.init_strategy == "sobol":
                    if not hasattr(_draw_candidates, "_sampler"):
                        _draw_candidates._sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
                    U = _draw_candidates._sampler.random(n_draw)

                else:
                    raise ValueError(f"Unknown init_strategy='{self.init_strategy}'")

                return qmc.scale(U, lo, hi)

            while len(X_out) < n_total:
                remaining = n_total - len(X_out)
                n_draw = max(remaining, 16)
                Xcand = _draw_candidates(n_draw)
                Xcand = _maybe_snap(Xcand)

                for i in range(Xcand.shape[0]):
                    k = _key_row(Xcand[i])
                    if k in seen:
                        continue
                    seen.add(k)
                    X_out.append(Xcand[i].copy())
                    if len(X_out) >= n_total:
                        break

            return np.asarray(X_out, dtype=float)

        if self.dataset is not None:
            X_cand = self.dataset.X
            y_cand = self.dataset.y
            N = int(X_cand.shape[0])

            self.compute_global_best_feasible_from_dataset()
            print("Global best feasible y over entire dataset:", self.global_best_feasible_y)
            print("Global best feasible x over entire dataset:", self.global_best_feasible_x)

            if X0 is not None:
                idx_user = np.asarray(X0)

                if idx_user.ndim == 1:
                    idx_user = idx_user.astype(int).ravel()
                elif idx_user.ndim == 2:
                    X_user = np.asarray(idx_user, dtype=float)
                    if X_user.shape[1] != self.dim:
                        raise ValueError(f"In dataset mode, X0 must be indices (1D) or points with shape (n,{self.dim}).")

                    map_ds = {}
                    for i in range(N):
                        map_ds[_key_row(X_cand[i])] = i

                    idx_list = []
                    for i in range(X_user.shape[0]):
                        k = _key_row(X_user[i])
                        if k not in map_ds:
                            raise ValueError(
                                "Could not match X0 rows exactly to dataset.X. "
                                "Please pass indices (1D integer array) in dataset mode."
                            )
                        idx_list.append(map_ds[k])
                    idx_user = np.asarray(idx_list, dtype=int)
                else:
                    raise ValueError("In dataset mode, X0 must be indices (1D) or points (2D).")

                if np.unique(idx_user).size != idx_user.size:
                    raise ValueError("Duplicate indices in X0 (dataset mode).")

                n_user = int(idx_user.size)
                if self.initial_points < n_user:
                    self.initial_points = n_user
                n_target = int(min(self.initial_points, N))

                idx_full = self.rng.choice(N, size=n_target, replace=False)

                if not np.array_equal(idx_full[:n_user], idx_user):
                    raise ValueError(
                        "To get extra points exactly identical to initialize(None), "
                        "X0 (indices) must be the prefix of the indices that initialize(None) would pick "
                        "with the same random_state and initial_points."
                    )
                else:
                    idx0 = idx_full
            else:
                n0 = min(int(self.initial_points), N)
                idx0 = self.rng.choice(N, size=n0, replace=False)

            self.used_idx[idx0] = True
            self.train_idx = np.asarray(idx0, dtype=int)
            self.X_train = np.asarray(X_cand[idx0], dtype=float)
            self.y_train = np.asarray(y_cand[idx0], dtype=float).ravel()
            self.G_train = np.asarray(self.dataset.G[idx0], dtype=float)

            if self.logger is not None:
                mask_all = self.feasibility_mask_idx(idx0)
                best_feas, _ = self.best_feasible_value()
                for i in range(self.X_train.shape[0]):
                    self.logger.log(
                        iter=0,
                        x_next=self.X_train[i],
                        y_next=self.y_train[i],
                        feasible=bool(mask_all[i]),
                        y_best_feasible=best_feas,
                    )
            return

        if X0 is not None:
            X0, y0, G0 = self._check_warmstart_shapes(X0, y0=y0, G0=G0)
            n_user = int(X0.shape[0])

            if self.initial_points < n_user:
                self.initial_points = n_user
            n_target = int(self.initial_points)

            X_full = _generate_full_initial_design(n_target)

            X0_snapped = _maybe_snap(X0)
            if (y0 is not None or G0 is not None) and (not np.all(X0_snapped == X0)):
                raise ValueError(
                    "X0 is not on the discrete grid but y0/G0 were provided. "
                    "Provide X0 already snapped, or pass y0=None and G0=None."
                )
            X0 = X0_snapped

            if np.unique([_key_row(X0[i]) for i in range(n_user)]).size != n_user:
                raise ValueError("Duplicate points in user-provided X0.")
            
            if not np.all(X_full[:n_user] == X0):
                raise ValueError(
                    "To get extra points exactly identical to initialize(None), "
                    "X0 must be exactly the first X0.shape[0] points produced by initialize(None) "
                    "(same random_state, init_strategy, discrete_values, and initial_points logic)."
                )

            X_extra = X_full[n_user:, :]
            self.X_train = X_full

            if y0 is None:
                y_all = self.objective_function(self.X_train)
                self.y_train = np.asarray(y_all, dtype=float).ravel()
            else:
                y0 = np.asarray(y0, dtype=float).ravel()
                y_extra = np.asarray(self.objective_function(X_extra), dtype=float).ravel() if X_extra.shape[0] > 0 else np.array([], dtype=float)
                self.y_train = np.concatenate([y0, y_extra])

            m = self._n_constraints()
            if m > 0:
                if G0 is None:
                    G_all = self._compute_constraints(self.X_train)
                    self.G_train = np.asarray(G_all, dtype=float)
                else:
                    G0 = np.asarray(G0, dtype=float)
                    if X_extra.shape[0] > 0:
                        G_extra = np.asarray(self._compute_constraints(X_extra), dtype=float)
                        self.G_train = np.vstack([G0, G_extra])
                    else:
                        self.G_train = np.asarray(G0, dtype=float)
            else:
                self.G_train = None

            if self.logger is not None:
                best_feas, mask_all = self.best_feasible_value()
                for i in range(self.X_train.shape[0]):
                    self.logger.log(
                        iter=0,
                        x_next=self.X_train[i],
                        y_next=self.y_train[i],
                        feasible=bool(mask_all[i]),
                        y_best_feasible=best_feas,
                    )
            return

        X_full = _generate_full_initial_design(int(self.initial_points))
        self.X_train = np.asarray(X_full, dtype=float)

        y0 = self.objective_function(self.X_train)
        self.y_train = np.asarray(y0, dtype=float).ravel()

        self.G_train = self._compute_constraints(self.X_train)

        if self.logger is not None:
            best_feas, mask_all = self.best_feasible_value()
            for i in range(self.X_train.shape[0]):
                self.logger.log(
                    iter=0,
                    x_next=self.X_train[i],
                    y_next=self.y_train[i],
                    feasible=bool(mask_all[i]),
                    y_best_feasible=best_feas,
                )

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
                X_pool = X_cand[idx_pool]
                pos = self.af.maximise_over_dataset(
                    gp=self.gp, X_cand=X_pool, y_best=y_best, return_index=True
                )
                idx_next = int(idx_pool[int(pos)])

            self.used_idx[idx_next] = True
            self.train_idx = np.append(self.train_idx, idx_next)

            x_next = X_cand[idx_next].reshape(1, -1)
            y_next = float(y_cand[idx_next])

            self.X_train = np.vstack([self.X_train, x_next])
            self.y_train = np.append(self.y_train, y_next)
            self.G_train = np.vstack([self.G_train, self.dataset.G[idx_next]])

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
            best_idx = None

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

        gp = GP(kernel_name="Matern", random_state=random_state, length_scale=1.0)()

        af = AF(
            kind="lcb",
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
            gp=gp,
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

    
    run_case(
        case_name="continuous",
        discrete_values=None,              # or [None, None]
        random_state=16
    )
    
    run_case(
        case_name="mixed_x2_discrete",
        discrete_values=[
            None,                          # x1 continuous
            np.linspace(0.0, 5.0, 11),     # x2 in {0.0, 0.5, 1.0, ..., 5.0}
        ],
        random_state=16
    )

    run_case(
        case_name="discrete_both",
        discrete_values=[
            np.linspace(0.0, 5.0, 11),     # x1 in {0.0, 0.5, ..., 5.0}
            np.linspace(0.0, 5.0, 11),     # x2 in {0.0, 0.5, ..., 5.0}
        ],
        random_state=16
    )
    
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
    random_state = 16

    gp = GP(kernel_name="Matern", random_state=random_state, length_scale=1.0)()

    af = AF(
        kind="ei",
        kappa=1.0,
        ml_on_bounds=True,
        ml_on_bounds_parameters={
            "name": "ridge",
            "task": "classification",
            "constraint_bounds": [(0.0, 2.1)],
        },
        ml_on_target=True,
        ml_on_target_parameters={
            "name": "nn",
            "task": "regression",
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500
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
        gp=gp,
        af=af,
        constraint_functions=[],     # constraints come from ds.G in dataset mode
        initial_points=11,
        random_state=random_state,
        logger=logger,
        dataset=ds,
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

    bo.initialize(X0=X_init)    # samples initial points from ds.X / ds.y

    x_best, y_best = bo.run(n_iterations=60, n_restarts=5, verbose=True)
    print("Best feasible x =", x_best, " y =", y_best)

    # CSV metrics naming
    bounds_metric = "mape"   # regression constraint models
    target_metric = ""       # no mlontarget here
    bo.logger.to_csv("ligen_log.csv", bounds_metric, target_metric)
    

if __name__ == "__main__":
    main()
