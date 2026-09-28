"""Statistical validation of metal_sim against the Brian2 reference outputs."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from brain_bridge import MetalBrain

BRAIN = Path.home() / "fly-brain" / "data" / "results"


def metrics(steps: np.ndarray, neurons: np.ndarray, p9_ids: np.ndarray,
            stim_ids: np.ndarray, seconds: float) -> dict:
    t_s = steps * 0.1 / 1000.0
    total = len(steps)
    active = len(set(neurons.tolist()))
    p9 = int(np.isin(neurons, p9_ids).sum())
    stim_spikes = int(np.isin(neurons, stim_ids).sum())
    stim_rate = stim_spikes / seconds / len(stim_ids)
    return {"total": total, "active": active, "p9": p9,
            "stim_rate_hz": stim_rate}


def brian2_metrics(parquet: Path, p9_ids: np.ndarray,
                   stim_ids: np.ndarray, seconds: float) -> dict:
    df = pd.read_parquet(parquet)
    idx = df["neuron_index"].to_numpy()
    return metrics(np.zeros(len(idx)), idx, p9_ids, stim_ids, seconds)


def show(name: str, m: dict) -> None:
    print(f"  {name:<12} spikes={m['total']:>7d}  active={m['active']:>4d}"
          f"  P9={m['p9']:>5d}  stim_rate={m['stim_rate_hz']:.0f}Hz")


b = MetalBrain()
print(f"loaded: n={b.n} syn={b.n_syn}")

sugar_idx = b.groups["sugar"]
p9_idx = np.concatenate([b.groups["p9_left"], b.groups["p9_right"]])

for exp, parquet in [("sugar", BRAIN / "brain_sugar_t1s.parquet"),
                     ("p9", BRAIN / "brain_p9_t1s.parquet")]:
    rate = 200.0 if exp == "sugar" else 100.0
    b.reset()
    if exp == "sugar":
        b.set_rate("sugar", rate)
    else:
        b.set_rate("p9_left", rate)
        b.set_rate("p9_right", rate)
    t0 = time.perf_counter()
    b.run(1.0)
    wall = time.perf_counter() - t0
    sp_t, sp_n = b.drain_spikes()
    print(f"[{exp}] sim=1.0s wall={wall:.2f}s realtime={1.0 / wall:.1f}x")
    print(f"  metal        {metrics(sp_t, sp_n, p9_idx, sugar_idx if exp == 'sugar' else p9_idx, 1.0)}")
    show("metal", metrics(sp_t, sp_n, p9_idx,
                          sugar_idx if exp == "sugar" else p9_idx, 1.0))
    show("brian2", brian2_metrics(parquet, p9_idx,
                                  sugar_idx if exp == "sugar" else p9_idx, 1.0))

b.close()
