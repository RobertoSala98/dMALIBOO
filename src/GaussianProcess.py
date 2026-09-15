from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

class GP:
    def __init__(self, kernel_name="Matern", random_state=None, **kernel_kwargs):
        self.kernel_name = kernel_name
        self.kernel_kwargs = kernel_kwargs
        self.random_state = random_state
        self.gpmodel = None

    def __call__(self, optimize_hyperparameters=True):
        kernel = self._get_kernel()

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            optimizer=(
                "fmin_l_bfgs_b"
                if optimize_hyperparameters
                else None
            ),
            n_restarts_optimizer=(
                5 if optimize_hyperparameters else 0
            ),
            random_state=self.random_state,
        )

    def _get_kernel(self):
        if self.kernel_name == 'RBF':
            return RBF(**self.kernel_kwargs)
        elif self.kernel_name == 'Matern':
            return Matern(**self.kernel_kwargs)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def clone_GP(
        self,
        length_scale=None,
        nu=None,
        optimize_hyperparameters=True,
    ):
        kwargs = dict(self.kernel_kwargs)

        if self.kernel_name == "RBF":
            if nu is not None:
                raise ValueError(
                    "nu is not valid for an RBF kernel."
                )

            if length_scale is not None:
                kwargs["length_scale"] = float(length_scale)

        elif self.kernel_name == "Matern":
            if length_scale is not None:
                kwargs["length_scale"] = float(length_scale)

            if nu is not None:
                kwargs["nu"] = float(nu)

        else:
            raise ValueError(
                f"Unknown kernel: {self.kernel_name}"
            )

        return GP(
            kernel_name=self.kernel_name,
            random_state=self.random_state,
            **kwargs,
        )(
            optimize_hyperparameters=optimize_hyperparameters
        )
    
def main():

    X_train = np.random.uniform(0, 5, size=(40, 2))
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.1 * np.random.randn(40)

    gp = GP(kernel_name='Matern', length_scale=2.0, nu=1.5)()
    gp.fit(X_train, y_train)

    x1 = np.linspace(0, 5, 50)
    x2 = np.linspace(0, 5, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])

    y_mean, y_std = gp.predict(X_test, return_std=True)

if __name__ == "__main__":
    main()