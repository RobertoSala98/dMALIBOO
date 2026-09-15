import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
filename = "resources/muDock.csv"
df = pd.read_csv(filename)

# Specify the column you want to analyze
column_name = "execution_time"

def plot_from_csv(file_path, x_col, y_col, filter_dict=None):
    # Load data
    df = pd.read_csv(file_path)

    # Optional filtering (e.g., fix some parameters)
    if filter_dict:
        for key, value in filter_dict.items():
            df = df[df[key] == value]

    # Sort for nicer plots (optional but recommended)
    df = df.sort_values(by=x_col)

    # Plot
    plt.figure()
    plt.scatter(df[x_col], df[y_col], marker='o')

    # Labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.ylim(0.0,0.15)
    plt.title(f"{y_col} vs {x_col}")

    plt.grid(True)
    plt.savefig("2D-plot.png")
    plt.close()

# Drop missing values (optional but recommended)
data = df[column_name].dropna()

# Plot histogram (distribution)
plt.figure()
plt.hist(data, bins=50)
plt.xlabel(column_name)
plt.ylabel("Frequency")
plt.title(f"Distribution of {column_name}")
plt.savefig("distribution_plot.png")
plt.close()


# Cumulative Distribution (CDF)
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

plt.figure()
plt.semilogx(sorted_data, cdf)
plt.plot([0.1,0.1], [0,1], 'b--')
plt.plot([5,5], [0,1], 'r--')
plt.plot([100,100], [0,1], 'g--')
plt.xlim(0.0,1000.0)
plt.ylim(0.0,0.55)
plt.xlabel(column_name)
plt.ylabel("Cumulative Probability")
plt.title(f"Cumulative Distribution of {column_name}")
plt.grid(True)
plt.savefig("cumulative_distribution_plot.png")
plt.close()

# mAP in [0.2, 1.0]
# min avg_time_per_sample
# energy_kwh in [0.0, 0.15]

plot_from_csv(filename, "median_energy", "execution_time", filter_dict=None)