import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

plt.figure(figsize=(10, 6))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_ax(ax: Axes, df: pd.DataFrame, title: str):
    for name, group in df.groupby("name"):
        x_values = [str(x) for x in group["M"]]
        ax.plot(x_values, group["TFLOPS"], marker="o", label=name)

    ax.set_xlabel("M/N")
    ax.set_ylabel("TFLOPS")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

plot_ax(ax1, pd.read_csv("sgemm_bench_square.csv", sep="|"), title="SGEMM Benchmark Square")
plot_ax(ax2, pd.read_csv("sgemm_bench_fixk.csv", sep="|"), title="SGEMM Benchmark Fixed K")

plt.tight_layout()
plt.savefig("sgemm_bench.png")
