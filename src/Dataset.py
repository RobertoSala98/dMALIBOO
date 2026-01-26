# Dataset.py
import os
import numpy as np

class Dataset:
    """
    File-backed dataset wrapper.

    Supports:
      - Loading from .csv/.tsv/.txt (pandas if available, else numpy.genfromtxt)
      - Loading from .npy and .npz (numpy.load)

    Column roles:
      - X columns: design variables
      - y column: objective
      - g columns: constraint values (one column per constraint)
      - t column: execution time (optional)

    Important:
      - x_cols must be disjoint from y_col and g_cols (inputs vs labels).
      - t_col is allowed to overlap with y_col or g_cols (aliasing is allowed).
      - By default, t_col cannot overlap with x_cols (can be relaxed).
    """

    def __init__(self, data, colnames,
                 x_cols=None, y_col=None, g_cols=None, t_col=None,
                 path=None, allow_time_in_x=False):

        self.path = path
        self.allow_time_in_x = bool(allow_time_in_x)

        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError("data must be a 2D array-like object.")

        self.colnames = list(colnames)
        if len(self.colnames) != self.data.shape[1]:
            raise ValueError("colnames length must match number of columns in data.")

        self.x_cols = x_cols
        self.y_col = y_col
        self.g_cols = g_cols or []
        self.t_col = t_col

        # Defaults: last column is y, all previous columns are X
        if self.x_cols is None and self.y_col is None:
            if self.data.shape[1] < 2:
                raise ValueError("Need at least 2 columns to infer X and y.")
            self.x_cols = list(range(self.data.shape[1] - 1))
            self.y_col = self.data.shape[1] - 1
        elif self.x_cols is None and self.y_col is not None:
            y_idx = self._col_to_index(self.y_col)
            self.x_cols = [i for i in range(self.data.shape[1]) if i != y_idx]
        elif self.x_cols is not None and self.y_col is None:
            raise ValueError("If x_cols is provided, y_col must also be provided.")

        self.x_idx = [self._col_to_index(c) for c in self.x_cols]
        self.y_idx = self._col_to_index(self.y_col)
        self.g_idx = [self._col_to_index(c) for c in self.g_cols]
        self.t_idx = None if self.t_col is None else self._col_to_index(self.t_col)

        self._validate_roles()


    @classmethod
    def from_file(cls, path,
                  x_cols=None, y_col=None, g_cols=None, t_col=None,
                  sep=None, npz_key=None,
                  csv_kwargs=None,
                  allow_time_in_x=False):
        """
        Parameters (common)
        -------------------
        path : str
            File path.

        x_cols, y_col, g_cols, t_col :
            Column specs as list[int]/list[str] or int/str (for y_col, t_col).

        allow_time_in_x : bool
            If True, allow t_col to also be an X column.

        Parameters (CSV/TSV/TXT)
        ------------------------
        sep : str or None
            Delimiter. If None: ',' for .csv else '\\t'.

        csv_kwargs : dict or None
            Passed to pandas.read_csv / numpy.genfromtxt as appropriate.
            Useful keys include: header, names, skiprows, comment, etc.

        Parameters (.npz)
        -----------------
        npz_key : str or None
            Key to select array inside .npz. If None and multiple arrays exist, raises.
        """
        ext = os.path.splitext(path)[1].lower()
        csv_kwargs = csv_kwargs or {}

        if ext in [".csv", ".tsv", ".txt"]:
            data, colnames = cls._load_text_table(path, sep=sep, csv_kwargs=csv_kwargs)
            return cls(data, colnames,
                       x_cols=x_cols, y_col=y_col, g_cols=g_cols, t_col=t_col,
                       path=path, allow_time_in_x=allow_time_in_x)

        if ext == ".npy":
            arr = np.load(path)
            if np.asarray(arr).ndim != 2:
                raise ValueError(".npy must contain a 2D array.")
            colnames = ["col_%d" % i for i in range(arr.shape[1])]
            return cls(arr, colnames,
                       x_cols=x_cols, y_col=y_col, g_cols=g_cols, t_col=t_col,
                       path=path, allow_time_in_x=allow_time_in_x)

        if ext == ".npz":
            z = np.load(path)
            keys = list(z.keys())
            if npz_key is None:
                if len(keys) != 1:
                    raise ValueError("npz_key must be provided (available keys: %s)" % keys)
                npz_key = keys[0]
            arr = z[npz_key]
            if np.asarray(arr).ndim != 2:
                raise ValueError(".npz selected array must be 2D.")
            colnames = ["col_%d" % i for i in range(arr.shape[1])]
            return cls(arr, colnames,
                       x_cols=x_cols, y_col=y_col, g_cols=g_cols, t_col=t_col,
                       path=path, allow_time_in_x=allow_time_in_x)

        raise ValueError("Unsupported file extension: %s" % ext)


    @staticmethod
    def _load_text_table(path, sep=None, csv_kwargs=None):
        csv_kwargs = csv_kwargs or {}

        if sep is None:
            sep = "," if path.lower().endswith(".csv") else "\t"

        try:
            import pandas as pd
            df = pd.read_csv(path, sep=sep, **csv_kwargs)
            return df.to_numpy(), df.columns.tolist()
        except ImportError:
            # Fallback: numpy.genfromtxt
            # Note: if the file has a header row, consider setting skip_header=1
            # via csv_kwargs: {"skip_header": 1} (numpy keyword).
            gen_kwargs = dict(csv_kwargs)
            # numpy uses 'delimiter', not 'sep'
            gen_kwargs.pop("sep", None)
            arr = np.genfromtxt(path, delimiter=sep, **gen_kwargs)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            colnames = ["col_%d" % i for i in range(arr.shape[1])]
            return arr, colnames


    def _col_to_index(self, c):
        if isinstance(c, int):
            if c < 0 or c >= self.data.shape[1]:
                raise IndexError("Column index out of range: %d" % c)
            return c
        if isinstance(c, str):
            if c not in self.colnames:
                raise KeyError("Unknown column name: %s" % c)
            return self.colnames.index(c)
        raise TypeError("Column spec must be int or str.")

    def _validate_roles(self):
        # Enforce that X / y / g are disjoint (inputs vs labels)
        used_core = self.x_idx + [self.y_idx] + self.g_idx
        if len(set(used_core)) != len(used_core):
            raise ValueError("x_cols, y_col, and g_cols must be disjoint.")

        # Allow time to alias y or any g (requested behavior)
        if self.t_idx is not None and (not self.allow_time_in_x) and (self.t_idx in self.x_idx):
            raise ValueError("t_col overlaps with x_cols; set allow_time_in_x=True to allow this.")


    @property
    def X(self):
        return np.asarray(self.data[:, self.x_idx], dtype=float)

    @property
    def y(self):
        return np.asarray(self.data[:, self.y_idx], dtype=float).reshape(-1)

    @property
    def G(self):
        if not self.g_idx:
            return None
        return np.asarray(self.data[:, self.g_idx], dtype=float)

    @property
    def t(self):
        # Optional: may alias y or one constraint column
        if self.t_idx is None:
            return None
        return np.asarray(self.data[:, self.t_idx], dtype=float).reshape(-1)

    @property
    def n(self):
        return int(self.data.shape[0])

    @property
    def dim(self):
        return int(len(self.x_idx))

def main():
    filename = "resources/oscarp.csv"

    x_cols = [
        "parallelism_ffmpeg-0",
        "parallelism_librosa",
        "parallelism_ffmpeg-1",
        "parallelism_ffmpeg-2",
        "parallelism_deepspeech",
    ]

    ds = Dataset.from_file(
        filename,
        x_cols=x_cols,
        y_col="cost",
        g_cols=["total_time"],
        t_col="total_time",
    )

    print("Loaded file:", ds.path)
    print("n =", ds.n)
    print("dim =", ds.dim)
    print("Columns:", ds.colnames)

    print("\nFirst row:")
    print("X[0] =", ds.X[0].tolist())
    print("y[0] =", float(ds.y[0]))
    print("t[0] =", None if ds.t is None else float(ds.t[0]))

    print("\nShapes:")
    print("X shape:", ds.X.shape)
    print("y shape:", ds.y.shape)
    print("G shape:", None if ds.G is None else ds.G.shape)
    print("t shape:", None if ds.t is None else ds.t.shape)

    # Consistency check: since g_cols == ["total_time"] and t_col == "total_time",
    # ds.t should match ds.G[:,0]
    if ds.G is not None and ds.t is not None:
        max_abs_diff = abs(ds.G[:, 0] - ds.t).max()
        print("\nCheck t == G[:,0]: max_abs_diff =", float(max_abs_diff))


if __name__ == "__main__":
    main()