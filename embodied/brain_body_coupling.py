"""Brain-body coupling replay: fly-brain P9 spike rate drives flygym CPG frequency.

Timeline (2s body time = 2x1s brain-time trials concatenated):
  0.0-1.0s  sugar GRNs stimulated (21 neurons @200Hz) -> P9 mostly quiet -> slow walk
  1.0-2.0s  P9 descending neurons stimulated (2 @100Hz) -> strong drive -> sprint
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

BRAIN_DIR = Path.home() / "fly-brain" / "data" / "results"
OUT_DIR = Path(__file__).parent / "demo_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P9_IDS = [720575940627652358, 720575940635872101]  # P9 left / right
BIN_SEC = 0.05
BASE_FREQ = 12.0
FREQ_MIN_MULT = 0.4
FREQ_MAX_MULT = 1.6


def p9_rate_curve(parquet: Path) -> np.ndarray:
    df = pd.read_parquet(parquet)
    spikes = df[df["flywire_id"].isin(P9_IDS)]["time_ms"].to_numpy() / 1000.0
    n_bins = int(np.ceil(spikes.max() / BIN_SEC)) if spikes.size else 20
    n_bins = max(n_bins, 20)
    counts, _ = np.histogram(spikes, bins=np.arange(n_bins + 1) * BIN_SEC)
    rate = counts / BIN_SEC / len(P9_IDS)  # Hz per neuron
    kernel = np.ones(3) / 3
    return np.convolve(rate, kernel, mode="same")


sugar_rate = p9_rate_curve(BRAIN_DIR / "brain_sugar_t1s.parquet")
p9_rate = p9_rate_curve(BRAIN_DIR / "brain_p9_t1s.parquet")
rate = np.concatenate([sugar_rate, p9_rate])
drive = np.clip(rate / rate.max(), 0, 1)
freq_mult = FREQ_MIN_MULT + (FREQ_MAX_MULT - FREQ_MIN_MULT) * drive
print(f"P9 rate: sugar max {sugar_rate.max():.0f} Hz, p9 max {p9_rate.max():.0f} Hz")
print(f"freq mult: {freq_mult.min():.2f} .. {freq_mult.max():.2f}")

# --- body side ---
fly = make_locomotion_fly(name="brain_body_fly", add_adhesion=True, colorize=True)
body_cam = fly.add_tracking_camera(
    name="body_cam",
    pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)),
    fovy=30.0,
)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
sim = Simulation(world)
renderer = sim.set_renderer([body_cam])

preprogrammed_steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)
base_freqs = cpg_network.intrinsic_freqs.copy()
controller = CPGController(
    cpg_network=cpg_network,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)

sim.reset()
initial_action = LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool),
)
apply_locomotion_action(sim, fly.name, initial_action)
sim.warmup()

run_time = len(freq_mult) * BIN_SEC
nsteps = int(run_time / sim.timestep)
thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
thorax_positions = np.full((nsteps, 3), np.nan, dtype=np.float32)

current_bin = -1
for step_idx in trange(nsteps, desc="Brain->body replay"):
    bin_idx = min(int(step_idx * sim.timestep / BIN_SEC), len(freq_mult) - 1)
    if bin_idx != current_bin:
        cpg_network.intrinsic_freqs = base_freqs * freq_mult[bin_idx]
        current_bin = bin_idx
    action = controller.step()
    apply_locomotion_action(sim, fly.name, action)
    sim.step_with_profile()
    thorax_positions[step_idx] = sim.get_body_positions(fly.name)[thorax_idx]
    sim.render_as_needed_with_profile()

print("Rendered frames:", len(renderer.frames[body_cam.name]))
sim.renderer.save_video(OUT_DIR / "brain_body_coupling.mp4")

time_grid = np.arange(len(freq_mult)) * BIN_SEC + BIN_SEC / 2
disp = thorax_positions[:, 0] - thorax_positions[0, 0]
fig, axes = plt.subplots(3, 1, figsize=(8, 6), tight_layout=True, sharex=True)
axes[0].bar(time_grid, rate, width=BIN_SEC * 0.9, color="darkred")
axes[0].axvline(1.0, color="k", ls="--", lw=1)
axes[0].set_ylabel("P9 rate (Hz)")
axes[0].set_title("Brain: P9 firing rate (sugar trial | P9 trial)")
axes[1].plot(time_grid, freq_mult * BASE_FREQ, color="darkorange", marker="o", ms=3)
axes[1].axvline(1.0, color="k", ls="--", lw=1)
axes[1].set_ylabel("CPG freq (Hz)")
axes[1].set_title("Bridge: P9 rate -> CPG frequency")
axes[2].plot(np.arange(nsteps) * sim.timestep, disp, color="navy")
axes[2].axvline(1.0, color="k", ls="--", lw=1)
axes[2].set_ylabel("Forward (mm)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Body: thorax displacement")
fig.savefig(OUT_DIR / "brain_body_coupling_diagnostics.png", dpi=150)
print("Saved:", OUT_DIR / "brain_body_coupling.mp4")
print(f"Total forward displacement: {disp[-1]:.1f} mm")
