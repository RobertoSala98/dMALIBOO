import numpy as np

from sklearn.utils.extmath import softmax
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class ML:
    def __init__(self, model_name, task="regression", poly_degree=2, random_state=None, **model_kwargs):
        """
        Parameters
        ----------
        model_name : str
            'ridge', 'rf', 'xgb', 'nn'
        task : str
            'regression' or 'classification'
        model_kwargs : dict
            Parameters passed to the underlying model
        """
        self.model_name = model_name.lower()
        self.task = task.lower()
        self.poly_degree = poly_degree
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.model = None

        self._constant_mode = False
        self._constant_class = None

        self._validate_inputs()
        self._build_model()

    def _validate_inputs(self):
        if self.task not in ["regression", "classification"]:
            raise ValueError("task must be 'regression' or 'classification'")

        if self.model_name not in ["ridge", "rf", "xgb", "nn"]:
            raise ValueError("model_name must be one of: ridge, rf, xgb, nn")
        
    def _build_model(self):

        if self.random_state is not None and "random_state" not in self.model_kwargs:
            self.model_kwargs["random_state"] = self.random_state

        if self.model_name == "ridge":
            base = Ridge(**self.model_kwargs) if self.task == "regression" else RidgeClassifier(**self.model_kwargs)
        elif self.model_name == "rf":
            base = RandomForestRegressor(**self.model_kwargs) if self.task == "regression" else RandomForestClassifier(**self.model_kwargs)
        elif self.model_name == "xgb":
            base = XGBRegressor(**self.model_kwargs) if self.task == "regression" else XGBClassifier(**self.model_kwargs)
        elif self.model_name == "nn":
            base = MLPRegressor(**self.model_kwargs) if self.task == "regression" else MLPClassifier(**self.model_kwargs)

        if self.poly_degree is not None and self.poly_degree > 1:
            self.model = Pipeline([
                ("poly", PolynomialFeatures(degree=self.poly_degree, include_bias=True)),
                ("est", base),
            ])
        else:
            self.model = base
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self._constant_mode = False
        self._constant_class = None

        if self.task == "classification":
            uniq = np.unique(y)
            if uniq.size < 2:
                # Don't train underlying sklearn model on a single class
                self._constant_mode = True
                self._constant_class = int(uniq[0])
                return self

        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)

        if self.task == "classification" and getattr(self, "_constant_mode", False):
            return np.full(X.shape[0], self._constant_class, dtype=int)

        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification")

        X = np.asarray(X)

        if getattr(self, "_constant_mode", False):
            # Neutral (does not distort AF too much when used as a multiplier)
            return np.tile(np.array([0.5, 0.5]), (X.shape[0], 1))

        if self.model_name == "ridge":
            scores = self.model.decision_function(X)  # works for Pipeline too
            if scores.ndim == 1:
                scores = np.c_[-scores, scores]
            return softmax(scores)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise RuntimeError("This model does not support predict_proba")

    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if self.task == "classification" and getattr(self, "_constant_mode", False):
            y_hat = np.full(y.shape[0], self._constant_class, dtype=int)
            return float(np.mean(y_hat == y))

        return self.model.score(X, y)
    
    def tune(self, X, y, param_grid, cv=5, scoring=None):
        grid = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X, y)
        self.model = grid.best_estimator_

        return grid.best_params_
    
    @staticmethod
    def mape(y_true, y_pred, eps=1e-12):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        denom = np.maximum(np.abs(y_true), eps)
        return float(np.mean(np.abs((y_true - y_pred) / denom)))

    def evaluate(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if self.task == "regression":
            y_hat = self.predict(X).ravel()
            return {
                "mape": self.mape(y, y_hat),
                "score": float(self.score(X, y)),
            }
        else:
            return {"accuracy": float(self.score(X, y))}


def main():
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=500, n_features=10, noise=0.1)

    ml = ML(
        model_name="rf",
        task="regression",
        n_estimators=200,
        random_state=0
    )

    ml.fit(X, y)
    y_pred = ml.predict(X)

    print("R² score:", ml.score(X, y))

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=0
    )

    ml = ML(
        model_name="xgb",
        task="classification",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False
    )

    ml = ML(model_name="xgb", task="classification", n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss", use_label_encoder=False)

    ml.fit(X, y)
    y_pred = ml.predict(X)
    y_prob = ml.predict_proba(X)

    print("Accuracy:", ml.score(X, y))

    ml = ML(
        model_name="nn",
        task="classification",
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=0
    )

    ml.fit(X, y)
    print("Accuracy:", ml.score(X, y))

if __name__ == "__main__":
    main()