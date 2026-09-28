"""Plot a spike raster from a fly-brain benchmark parquet output."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(parquet_path: str) -> str:
    df = pd.read_parquet(parquet_path)
    out_png = str(Path(parquet_path).with_suffix(".raster.png"))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(df["time_ms"], df["neuron_index"], s=6, c="crimson", alpha=0.7)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron index")
    exp = df["exp_name"].iloc[0] if "exp_name" in df else "spikes"
    n_neurons = df["neuron_index"].nunique()
    ax.set_title(f"{exp} — {len(df)} spikes / {n_neurons} neurons")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(out_png)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: plot_raster.py <spikes.parquet>")
    print(main(sys.argv[1]))
