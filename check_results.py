import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_mean_std(folder):
    y_best_feasible = []

    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)

            df = pd.read_csv(file_path)
            y_best_feasible.append(df["y_best_feasible"].iloc[-1])

    y_best_feasible_mean = np.mean(y_best_feasible)
    y_best_feasible_std = np.std(y_best_feasible, ddof=1)

    return y_best_feasible_mean, y_best_feasible_std


def extract_time_timeline(folder):
    """Compute mean and standard deviation of y_best_feasible over elapsed time."""
    dfs = []
    repetition_id = 0

    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)

            df = pd.read_csv(file_path)
            df["elapsed_time"] = (
                df["step_time"].fillna(0.0)
                + df["evaluation_time"].fillna(0.0)
            ).cumsum()
            df["repetition"] = repetition_id
            repetition_id += 1

            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Shared time grid: every elapsed time observed in any repetition
    time_grid = np.sort(data["elapsed_time"].unique())

    aligned = []

    for repetition, group in data.groupby("repetition", sort=False):
        group = (
            group[["elapsed_time", "y_best_feasible"]]
            .drop_duplicates(subset="elapsed_time", keep="last")
            .sort_values("elapsed_time")
            .set_index("elapsed_time")
        )

        # Carry forward the current best-so-far value
        y = group["y_best_feasible"].reindex(time_grid).ffill()
        y = y.dropna()

        aligned.append(
            pd.DataFrame(
                {
                    "elapsed_time": y.index,
                    "repetition": repetition,
                    "y_best_feasible": y.to_numpy(),
                }
            )
        )

    aligned = pd.concat(aligned, ignore_index=True)

    summary = (
        aligned
        .groupby("elapsed_time", as_index=False)["y_best_feasible"]
        .agg(
            y_best_feasible_mean="mean",
            y_best_feasible_std="std",
            n_repetitions="count",
        )
    )

    return summary


def extract_iteration_timeline(folder):
    """Compute mean and standard deviation of y_best_feasible over iterations."""
    dfs = []
    repetition_id = 0

    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)

            df = pd.read_csv(file_path)
            df["repetition"] = repetition_id
            repetition_id += 1

            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Shared iteration grid: every iteration observed in any repetition
    iteration_grid = np.sort(data["iter"].unique())

    aligned = []

    for repetition, group in data.groupby("repetition", sort=False):
        group = (
            group[["iter", "y_best_feasible"]]
            .drop_duplicates(subset="iter", keep="last")
            .sort_values("iter")
            .set_index("iter")
        )

        # Carry forward the current best-so-far value
        y = group["y_best_feasible"].reindex(iteration_grid).ffill()
        y = y.dropna()

        aligned.append(
            pd.DataFrame(
                {
                    "iter": y.index,
                    "repetition": repetition,
                    "y_best_feasible": y.to_numpy(),
                }
            )
        )

    aligned = pd.concat(aligned, ignore_index=True)

    summary = (
        aligned
        .groupby("iter", as_index=False)["y_best_feasible"]
        .agg(
            y_best_feasible_mean="mean",
            y_best_feasible_std="std",
            n_repetitions="count",
        )
    )

    return summary


def plot_summaries(
    summaries,
    x_column="iter",
    xlabel="Iteration",
    ylabel=r"$y^{\mathrm{best}}_{\mathrm{feasible}}$",
    output_path="comparison.png",
):
    """Overlay multiple timeline summaries in the same plot.

    `summaries` is a list of dictionaries:
    [
        {"label": "LHS", "summary": summary_lhs},
        {"label": "Random", "summary": summary_random},
    ]
    """
    plt.figure(figsize=(8, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, item in enumerate(summaries):
        summary = item["summary"]
        label = item["label"]
        color = colors[i % len(colors)]

        x = summary[x_column]
        mean = summary["y_best_feasible_mean"]
        std = summary["y_best_feasible_std"]

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        plt.plot(
            x,
            mean,
            color=color,
            linewidth=2,
            label=label,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def main():

    main_folder = "output/schwefel/kernel_AF"

    # folders = {
    #     "LHS": os.path.join(main_folder, "lhs"),
    #     "Random": os.path.join(main_folder, "random"),
    #     "Sobol": os.path.join(main_folder, "sobol")
    # }

    folders = {
        "Matern + EI": os.path.join(main_folder, "Matern_EI"),
        "Matern + UCB": os.path.join(main_folder, "Matern_LCB"),
        "RBF + EI": os.path.join(main_folder, "RBF_EI"),
        "RBF + UCB": os.path.join(main_folder, "RBF_LCB"),
    }

    # Plot versus number of iterations
    iteration_summaries = [
        {"label": label, "summary": extract_iteration_timeline(folder)}
        for label, folder in folders.items()
    ]

    plot_summaries(
        iteration_summaries,
        x_column="iter",
        xlabel="Iteration",
        output_path=os.path.join(main_folder, "comparison_iterations.png"),
    )

    # Plot versus elapsed time
    time_summaries = [
        {"label": label, "summary": extract_time_timeline(folder)}
        for label, folder in folders.items()
    ]

    plot_summaries(
        time_summaries,
        x_column="elapsed_time",
        xlabel="Elapsed time [s]",
        output_path=os.path.join(main_folder, "comparison_time.png"),
    )


if __name__ == "__main__":
    main()