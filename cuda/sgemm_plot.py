import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

def plot_figure():
    plt.figure(figsize=(10, 6))
    df = pd.read_csv("output/hgemm_bench_square.csv", sep="|")
    xlabel = "M/N/K"
    ylabel = "TFLOPS"
    title = "HGEMM Benchmark"
    save_path = "hgemm_bench.png"

    for name, group in df.groupby("name"):
        x_values = [str(x) for x in group["M"]]
        plt.plot(x_values, group["TFLOPS"], marker="o", label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_multifig():
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

    plot_ax(ax1, pd.read_csv("output/hgemm_bench_square.csv", sep="|"), title="HGEMM Benchmark Square")
    plot_ax(ax2, pd.read_csv("output/hgemm_bench_fixk.csv", sep="|"), title="HGEMM Benchmark Fixed K")

    plt.tight_layout()
    plt.savefig("hgemm_bench.png")


plot_figure()
