"""Interactive window with brain-body coupling replay.

The fly walks driven by the P9 firing-rate curve measured from the fly-brain
simulations (sugar trial -> P9 silent -> slow; P9 trial -> 117 Hz -> sprint).
The 2s timeline loops continuously. Ctrl+drag to poke the fly.
"""
import mujoco
import numpy as np
import pandas as pd
from pathlib import Path

from flygym import Simulation
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
P9_IDS = [720575940627652358, 720575940635872101]
BIN_SEC = 0.05
BASE_FREQ = 12.0
FREQ_MIN_MULT = 0.4
FREQ_MAX_MULT = 1.6


def p9_rate_curve(parquet: Path) -> np.ndarray:
    df = pd.read_parquet(parquet)
    spikes = df[df["flywire_id"].isin(P9_IDS)]["time_ms"].to_numpy() / 1000.0
    n_bins = max(int(np.ceil(spikes.max() / BIN_SEC)) if spikes.size else 20, 20)
    counts, _ = np.histogram(spikes, bins=np.arange(n_bins + 1) * BIN_SEC)
    rate = counts / BIN_SEC / len(P9_IDS)
    return np.convolve(rate, np.ones(3) / 3, mode="same")


rate = np.concatenate([
    p9_rate_curve(BRAIN_DIR / "brain_sugar_t1s.parquet"),
    p9_rate_curve(BRAIN_DIR / "brain_p9_t1s.parquet"),
])
freq_mult = FREQ_MIN_MULT + (FREQ_MAX_MULT - FREQ_MIN_MULT) * np.clip(rate / rate.max(), 0, 1)
timeline_sec = len(freq_mult) * BIN_SEC

fly = make_locomotion_fly(name="coupling_fly", add_adhesion=True, colorize=True)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
sim = Simulation(world)

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

print(
    f"Brain-driven walking, {timeline_sec:.0f}s loop: slow (sugar, P9 silent) "
    "then sprint (P9 active). Ctrl+drag to poke. Esc quits."
)
steps_per_frame = 10
step_idx = 0
current_bin = -1
with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    while viewer.is_running():
        sim_time = (step_idx * sim.timestep) % timeline_sec
        bin_idx = min(int(sim_time / BIN_SEC), len(freq_mult) - 1)
        if bin_idx != current_bin:
            cpg_network.intrinsic_freqs = base_freqs * freq_mult[bin_idx]
            current_bin = bin_idx
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        step_idx += 1
        if step_idx % steps_per_frame == 0:
            viewer.sync()
