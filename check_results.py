import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


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


def extract_time_timeline(folder, optimal_value):
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

    summary['mapr'] = (summary["y_best_feasible_mean"] - optimal_value)/abs(optimal_value)
    summary['mapr_std'] = summary["y_best_feasible_std"]/abs(optimal_value)

    return summary


def extract_iteration_timeline(folder, optimal_value):
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

    summary['mapr'] = (summary["y_best_feasible_mean"] - optimal_value)/abs(optimal_value)
    summary['mapr_std'] = summary["y_best_feasible_std"]/abs(optimal_value)

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

        final_x = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        final_y = mean.iloc[-1] if hasattr(mean, "iloc") else mean[-1]

        plt.annotate(
            f"{final_y:.2f}",
            xy=(final_x, final_y),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_summaries_mapr(
    summaries,
    x_column="iter",
    xlabel="Iteration",
    ylabel=r"$\mathrm{MAPR}$",
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
        mean = summary["mapr"]
        std = summary["mapr_std"]

        plt.fill_between(
            x,
            np.maximum(mean - std, 0),
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

        final_x = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        final_y = mean.iloc[-1] if hasattr(mean, "iloc") else mean[-1]

        plt.annotate(
            f"{final_y*100:.2f} %",
            xy=(final_x, final_y),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_results(main_folder, folders, optimal_value):
    # Plot versus number of iterations
    iteration_summaries = [
        {"label": label, "summary": extract_iteration_timeline(folder, optimal_value)}
        for label, folder in folders.items()
    ]

    plot_summaries(
        iteration_summaries,
        x_column="iter",
        xlabel="Iteration",
        output_path=os.path.join(main_folder, "comparison_iterations.png"),
    )

    plot_summaries_mapr(
        iteration_summaries,
        x_column="iter",
        xlabel="Iteration",
        output_path=os.path.join(main_folder, "comparison_iterations_mapr.png"),
    )

    # Plot versus elapsed time
    time_summaries = [
        {"label": label, "summary": extract_time_timeline(folder, optimal_value)}
        for label, folder in folders.items()
    ]

    plot_summaries(
        time_summaries,
        x_column="elapsed_time",
        xlabel="Elapsed time [s]",
        output_path=os.path.join(main_folder, "comparison_time.png"),
    )

    plot_summaries_mapr(
        time_summaries,
        x_column="elapsed_time",
        xlabel="Elapsed time [s]",
        output_path=os.path.join(main_folder, "comparison_time_mapr.png"),
    )


def main():

    # best_values = {
    #     4: 483.53250582561077,
    #     8: 784.7950797097578,
    #     12: 1716.9996061042643,
    #     16: 2282.6068412213044
    # }

    # d = 8

    # main_folder = "output/schwefel/initialization_strategy"
    # folders = {
    #     "LHS": os.path.join(main_folder, "lhs"),
    #     "Random": os.path.join(main_folder, "random"),
    #     "Sobol": os.path.join(main_folder, "sobol")
    # }
    # plot_results(main_folder, folders, best_values[d])

    # main_folder = "output/schwefel/kernel_AF"
    # folders = {
    #     "Matern + EI": os.path.join(main_folder, "Matern_EI"),
    #     "Matern + LCB": os.path.join(main_folder, "Matern_LCB"),
    #     "RBF + EI": os.path.join(main_folder, "RBF_EI"),
    #     "RBF + LCB": os.path.join(main_folder, "RBF_LCB"),
    # }
    # plot_results(main_folder, folders, best_values[d])

    # main_folder = "output/schwefel/ML_approach"
    # folders = {
    #     "Indicator-Indicator": os.path.join(main_folder, "indicator_indicator"),
    #     "Indicator-Probability": os.path.join(main_folder, "indicator_probability"),
    #     "Probability-Indicator": os.path.join(main_folder, "probability_indicator"),
    #     "Probability-Probability": os.path.join(main_folder, "probability_probability")
    # }
    # plot_results(main_folder, folders, best_values[d])

    # # oscarp
    # constr_values = [150, 300, 450, 600]
    # opt_values = [0.174361111, 0.174361111, 0.174361111, 0.174361111]

    # for id in range(len(constr_values)):
    #     constr_value = constr_values[id]
    #     opt_value = opt_values[id]
    #     path = f"output/oscarp/results/{constr_value}"
    #     folder = Path(path)
    #     folder.mkdir(parents=True, exist_ok=True)

    #     folders = {
    #         "Matern + EI": f"output/oscarp/Matern_EI/{constr_value}",
    #         "Matern + LCB": f"output/oscarp/Matern_lcb/{constr_value}",
    #         "RBF + EI": f"output/oscarp/RBF_ei/{constr_value}",
    #         "RBF + LCB": f"output/oscarp/RBF_lcb/{constr_value}",
    #     }

    #     plot_results(path, folders, opt_value)

    # query52
    constr_values = [262887, 334000, 407000, 645000, 2250000]
    opt_values = [682215, 682215, 682215, 682215, 682215]

    for id in range(len(constr_values)):
        constr_value = constr_values[id]
        opt_value = opt_values[id]
        path = f"output/query52/results/{constr_value}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        folders = {
            "Matern + EI": f"output/query52/Matern_ei/{constr_value}",
            "Matern + LCB": f"output/query52/Matern_lcb/{constr_value}",
            "RBF + EI": f"output/query52/RBF_ei/{constr_value}",
            "RBF + LCB": f"output/query52/RBF_lcb/{constr_value}",
        }

        plot_results(path, folders, opt_value)

    # stereomatch
    constr_values = [8000, 10000, 12000, 17000, 20000, 40000]
    opt_values = [50776.0, 45525.0, 40196.0, 40196.0, 37412.0, 36791.0]

    for id in range(len(constr_values)):
        constr_value = constr_values[id]
        opt_value = opt_values[id]
        path = f"output/stereomatch/results/{constr_value}"
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        folders = {
            "Matern + EI": f"output/stereomatch/Matern_ei/{constr_value}",
            "Matern + LCB": f"output/stereomatch/Matern_lcb/{constr_value}",
            "RBF + EI": f"output/stereomatch/RBF_ei/{constr_value}",
            "RBF + LCB": f"output/stereomatch/RBF_lcb/{constr_value}",
        }

        plot_results(path, folders, opt_value)

    # # ligen
    # constr_values = [2.0, 2.1, 2.2, 2.45, 2.75]
    # opt_values = [567.312555400384, 567.312555400384, 464.349441178703, 424.355184049556, 340.099251789615]

    # for id in range(len(constr_values)):
    #     constr_value = constr_values[id]
    #     opt_value = opt_values[id]
    #     path = f"output/ligen/results/{constr_value}"
    #     folder = Path(path)
    #     folder.mkdir(parents=True, exist_ok=True)

    #     folders = {
    #         "Matern + EI": f"output/ligen/Matern_ei/{constr_value}",
    #         "Matern + LCB": f"output/ligen/Matern_lcb/{constr_value}",
    #         "RBF + EI": f"output/ligen/RBF_ei/{constr_value}",
    #         "RBF + LCB": f"output/ligen/RBF_lcb/{constr_value}",
    #     }

    #     plot_results(path, folders, opt_value)

if __name__ == "__main__":
    main()