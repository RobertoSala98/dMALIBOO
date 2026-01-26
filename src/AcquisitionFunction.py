import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from sklearn.exceptions import ConvergenceWarning

from GaussianProcess import *
from MachineLearning import *

class AF:
    def __init__(self, kind="ei", ml_on_bounds=False, ml_on_target=False, random_state=None, **af_kwards):
        self.kind = kind

        if self.kind not in ["ei", "lcb"]:
            raise ValueError("AF not supported")

        self.af_kwards = af_kwards
        self.ml_on_bounds = ml_on_bounds
        self.ml_on_target = ml_on_target
        self.rng = np.random.default_rng(random_state)

        if self.ml_on_bounds:
            if "ml_on_bounds_parameters" in self.af_kwards:
                self.ml_on_bounds_parameters = self.af_kwards["ml_on_bounds_parameters"]
                other_params = {k: v for k, v in self.ml_on_bounds_parameters.items() if k not in ["name", "task", "constraint_bounds"]}
                other_params.setdefault("random_state", random_state)

                self.ml_bounds = []
                for constraint in range(len(self.ml_on_bounds_parameters["constraint_bounds"])):
                    self.ml_bounds.append(ML(self.ml_on_bounds_parameters["name"], self.ml_on_bounds_parameters["task"], **other_params))
            else:
                raise ValueError("You must provide ml_on_bounds_parameters")
            
        if self.ml_on_target:
            if "ml_on_target_parameters" in self.af_kwards:
                self.ml_on_target_parameters = self.af_kwards["ml_on_target_parameters"]

                other_params = {k: v for k, v in self.ml_on_target_parameters.items() if k not in ["name", "task"]}
                other_params.setdefault("random_state", random_state)

                self.ml_target = ML(self.ml_on_target_parameters["name"], self.ml_on_target_parameters["task"], **other_params)
            else:
                raise ValueError("You must provide ml_on_target_parameters")
        
        self.normalize_AF = False
        if self.kind == "lcb":
            if "bounds" in self.af_kwards:
                self.bounds = af_kwards["bounds"]
            else:
                raise ValueError("You need to provide bounds to the lcb AF")
            
            if (self.ml_on_bounds and self.ml_on_bounds_parameters["task"] == "classification") or (self.ml_on_target and self.ml_on_target_parameters["task"] == "classification"):
                self.normalize_AF = True

    def __call__(self, X, gp, y_best=None):
        
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        if self.kind == "ei":
            if y_best is None:
                raise ValueError("y_best must be provided for EI")
            af_value = self._ei(mu, sigma, y_best)
        else:
            kappa = self.af_kwards.get("kappa", 1.0)
            af_value = self._lcb(mu, sigma, kappa)

        if self.normalize_AF:
            af_value = np.maximum((af_value - self.min_AF) / (self.max_AF - self.min_AF), 0.0)

        if self.ml_on_bounds:
            if self.ml_on_bounds_parameters["task"] == 'regression':
                indicator = np.ones(X.shape[0], dtype=bool)
                for i, model in enumerate(self.ml_bounds):
                    value = model.predict(X).ravel()
                    lb, ub = self.ml_on_bounds_parameters["constraint_bounds"][i]
                    indicator &= (value >= lb) & (value <= ub)

                af_value[~indicator] = 0

            else:
                probabilities = np.ones(X.shape[0])
                for i, model in enumerate(self.ml_bounds):
                    value = model.predict_proba(X)[:, 1]
                    probabilities *= value
                af_value *= probabilities

        if self.ml_on_target:
            if self.ml_on_target_parameters["task"] == 'regression':
                value = self.ml_target.predict(X).ravel()
                af_value *= (value <= y_best)

            else: 
                value = self.ml_target.predict_proba(X)[:, 1]
                af_value *= value

        return af_value

    def _ei(self, mu, sigma, y_best):
        improvement = y_best - mu
        Z = improvement / sigma
        return improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def _lcb(self, mu, sigma, kappa=1.0):
        return -(mu - kappa * sigma)
    
    def maximise(self, gp, bounds, y_best=None, n_restarts=10):
        bounds = np.asarray(bounds)

        if self.normalize_AF:

            def est_min_AF(X_):
                x = np.atleast_2d(X_)
                mu, sigma = gp.predict(x, return_std=True)
                return self._lcb(mu, sigma)
            
            def est_max_AF(X_):
                x = np.atleast_2d(X_)
                mu, sigma = gp.predict(x, return_std=True)
                return -self._lcb(mu, sigma)
            
            self.min_AF = np.inf
            self.max_AF = -np.inf
            for _ in range(10):
                x0 = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
                
                trial_min_AF = minimize(est_min_AF, x0=x0, bounds=self.bounds, method="L-BFGS-B").fun
                trial_max_AF = minimize(est_max_AF, x0=x0, bounds=self.bounds, method="L-BFGS-B").fun

                self.min_AF = min(self.min_AF, trial_min_AF)
                self.max_AF = max(self.max_AF, trial_max_AF)

        def objective(x):
            if np.any(np.isnan(x)):
                return np.inf

            x = np.atleast_2d(x)
            return -self(x, gp, y_best=y_best)

        best_x = None
        best_val = np.inf
        restarts = 0

        # Could be a deadlock, check
        while restarts <= n_restarts or best_val == np.inf:
            x0 = self.rng.uniform(bounds[:, 0], bounds[:, 1])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")

            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

            restarts += 1

        return best_x
    
    def maximise_over_dataset(self, gp, X_cand, y_best=None, batch=1, return_index=False):
        X_cand = np.asarray(X_cand)

        if self.normalize_AF:
            mu, sigma = gp.predict(X_cand, return_std=True)

            base = self._lcb(mu, sigma, self.af_kwards.get("kappa", 1.0))

            self.min_AF = float(np.min(base))
            self.max_AF = float(np.max(base))

        vals = self(X_cand, gp=gp, y_best=y_best).ravel()

        if batch == 1:
            j = int(np.argmax(vals))
            return j if return_index else X_cand[j]

        idx = np.argsort(vals)[-batch:][::-1]
        return idx if return_index else X_cand[idx]

def main():
    def objective(X):
        x1, x2 = X[:, 0], X[:, 1]
        return np.sin(x1) + np.cos(x2) + 0.1 * (x1 - 2) ** 2

    np.random.seed(42)

    bounds = np.array([[0.0, 5.0],
                       [0.0, 5.0]])

    def sample_uniform(n):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2))

    # Constraints
    def g1(X): return X[:, 0] + X[:, 1]
    def g2(X): return X[:, 0] * X[:, 1]
    constraint_functions = [g1, g2]

    gp = GP(kernel_name="Matern", length_scale=1.0)()
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
            "task": "classification"
        },
        bounds = bounds
    )

    def feasibility_mask(X):
        mask = np.ones(X.shape[0], dtype=bool)
        for j, constr_fun in enumerate(constraint_functions):
            v = constr_fun(X).ravel()
            lb, ub = af.ml_on_bounds_parameters["constraint_bounds"][j]
            mask &= (v >= lb) & (v <= ub)
        return mask

    def best_feasible_value(X, y):
        mask = feasibility_mask(X)
        if not np.any(mask):
            return None, mask
        feas_idx = np.where(mask)[0]
        best_idx = feas_idx[np.argmin(y[feas_idx])]
        return float(y[best_idx]), mask

    def fit_constraint_models(X):
        task = af.ml_on_bounds_parameters["task"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for j, constr_fun in enumerate(constraint_functions):
                v = constr_fun(X).ravel()
                lb, ub = af.ml_on_bounds_parameters["constraint_bounds"][j]
                if task == "classification":
                    y_feas = ((v >= lb) & (v <= ub)).astype(int)
                    af.ml_bounds[j].fit(X, y_feas)
                else:
                    af.ml_bounds[j].fit(X, v)

    def fit_target_models(X, y, y_best):
        task = af.ml_on_target_parameters["task"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if task == "classification":
                y_feas = (y <= y_best).astype(int)
                af.ml_target.fit(X, y_feas)
            else:
                af.ml_target.fit(X, y)

    X_train = sample_uniform(10)
    y_train = objective(X_train)

    n_iterations = 20
    for i in range(n_iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X_train, y_train)

        best_feas, _ = best_feasible_value(X_train, y_train)
        y_best = best_feas if best_feas is not None else float(np.max(y_train))

        if af.ml_on_bounds:
            fit_constraint_models(X_train)
        if af.ml_on_target:
            fit_target_models(X_train, y_train, y_best)

        x_next = af.maximise(gp=gp, bounds=bounds, y_best=y_best, n_restarts=10).reshape(1, -1)
        y_next = objective(x_next)

        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

        best_feas, _ = best_feasible_value(X_train, y_train)
        print(
            f"Iter {i+1:02d} | "
            f"x_next = {x_next.ravel()} | "
            f"y_next = {y_next[0]:.4f} | "
            f"best_feasible = {best_feas if best_feas is not None else np.nan}"
        )

    best_feas, mask = best_feasible_value(X_train, y_train)
    if best_feas is not None:
        feas_idx = np.where(mask)[0]
        best_idx = feas_idx[np.argmin(y_train[feas_idx])]
    else:
        best_idx = np.argmin(y_train)

    print("\nBest solution found:")
    print("x =", X_train[best_idx])
    print("y =", y_train[best_idx])

if __name__ == "__main__":
    main()