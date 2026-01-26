import numpy as np
import json
import csv

class Logger:
    """
    Independent logger for BO runs.
    Stores one dict per iteration, extensible via `extra` fields.
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self.rows = []

    def log(self, iter, x_next, y_next, feasible=None, y_best_feasible=None, accuracy_ml_bounds=None, accuracy_ml_target=None):
        x_next = np.asarray(x_next).ravel()
        if x_next.size != self.dim:
            raise ValueError(f"x_next must have length {self.dim}, got {x_next.size}")

        row = {
            "iter": int(iter),
            "x": x_next.astype(float).tolist(),
            "y": float(np.asarray(y_next).ravel()[0]),
            "feasible": None if feasible is None else bool(feasible),
            "y_best_feasible": None if y_best_feasible is None else float(y_best_feasible),
            "accuracy_ml_bounds": None if accuracy_ml_bounds is None else accuracy_ml_bounds,
            "accuracy_ml_target": None if accuracy_ml_target is None else accuracy_ml_target
        }

        self.rows.append(row)

    def X(self):
        return np.asarray([r["x"] for r in self.rows], dtype=float)

    def y(self):
        return np.asarray([r["y"] for r in self.rows], dtype=float)

    def feasible_mask(self):
        return np.asarray([r["feasible"] for r in self.rows], dtype=object)

    def __len__(self):
        return len(self.rows)

    def to_json(self, path, indent=2):
        with open(path, "w") as f:
            json.dump(self.rows, f, indent=indent)

    def to_csv(self, path, bounds_metric, target_metric):
        if not self.rows:
            raise ValueError("No rows to log")
        
        n_constraints = len(self.rows[-1]["accuracy_ml_bounds"])
        
        fieldnames = (
            ["iter"] + 
            [f"x{i}" for i in range(self.dim)] + 
            ["y", "feasible", "y_best_feasible"] + 
            [f"{bounds_metric}_ml_bounds_{i+1}" for i in range(n_constraints)] + 
            [f"{target_metric}_ml_target"]
        )
        
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.rows:
                flat = {k: r.get(k, None) for k in fieldnames}
                for i, xi in enumerate(r["x"]):
                    flat[f"x{i}"] = xi
                if r.get("accuracy_ml_bounds") is not None:
                    for i, xi in enumerate(r["accuracy_ml_bounds"]):
                        flat[f"{bounds_metric}_ml_bounds_{i+1}"] = xi
                if r.get("accuracy_ml_target") is not None:
                    flat[f"{target_metric}_ml_target"] = r["accuracy_ml_target"]
                w.writerow(flat)
