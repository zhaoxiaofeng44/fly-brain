"""Fly playground: interactive candy + stick.

Keys (click the MuJoCo window first):
  C - drop a candy ~12mm ahead of the fly (fly stops to eat when close, eats in ~2.5s)
  S - stick strike from a random side (fly spins away and sprints = escape)
  Esc - quit

Candy/stick are non-colliding mocap props; their effects are brain-driven:
sugar proximity -> feeding stance; stick threat -> P9-style escape sprint.
"""
import mujoco as mj
import numpy as np
from pathlib import Path

import pandas as pd

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
P9_IDS = [720575940627652358, 720575940635872101]
BIN_SEC = 0.05
BASE_FREQ = 12.0

rng = np.random.default_rng(7)


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ v


def p9_rate_curve(parquet: Path) -> np.ndarray:
    df = pd.read_parquet(parquet)
    spikes = df[df["flywire_id"].isin(P9_IDS)]["time_ms"].to_numpy() / 1000.0
    n_bins = max(int(np.ceil(spikes.max() / BIN_SEC)) if spikes.size else 20, 20)
    counts, _ = np.histogram(spikes, bins=np.arange(n_bins + 1) * BIN_SEC)
    rate = counts / BIN_SEC / len(P9_IDS)
    return np.convolve(rate, np.ones(3) / 3, mode="same")


# baseline walking speed from the measured P9 curve (steady cruise)
p9_rate = p9_rate_curve(BRAIN_DIR / "brain_p9_t1s.parquet")
cruise_mult = float(np.clip(p9_rate.mean() / p9_rate.max(), 0.3, 1.0)) * 0.9

# --- world with candy and stick props (non-colliding mocap) ---
fly = make_locomotion_fly(name="play_fly", add_adhesion=True, colorize=True)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))

candy_body = world.mjcf_root.worldbody.add_body(name="candy", pos=[0, 0, -50])
candy_body.mocap = True
candy_body.add_geom(
    type=mj.mjtGeom.mjGEOM_SPHERE, size=[1.2, 0, 0], rgba=[1.0, 0.55, 0.1, 0.95],
    contype=0, conaffinity=0,
)
stick_body = world.mjcf_root.worldbody.add_body(name="stick", pos=[0, 0, -50])
stick_body.mocap = True
stick_body.add_geom(
    type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.5, 9, 0], rgba=[0.45, 0.3, 0.15, 0.95],
    contype=0, conaffinity=0,
)

body_cam = fly.add_tracking_camera(
    name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
    rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0,
)
sim = Simulation(world)
model, data = sim.mj_model, sim.mj_data

candy_mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "candy")]
stick_mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "stick")]
HIDDEN = np.array([0.0, 0.0, -50.0])

# --- controller ---
preprogrammed_steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep, intrinsic_amplitude=1.0,
    coupling_strength=10.0, convergence_coef=20.0, seed=0,
)
base_freqs = cpg_network.intrinsic_freqs.copy()
controller = CPGController(
    cpg_network=cpg_network, preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)

sim.reset()
initial_action = LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool),
)
apply_locomotion_action(sim, fly.name, initial_action)
sim.warmup()

# --- interaction state ---
candy_active = False
feed_time = 0.0
stick_t0 = None
stick_from = stick_to = None
escape_until = -1.0
escape_ds = np.array([1.0, 1.0])

thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))


def thorax_pos() -> np.ndarray:
    return np.asarray(sim.get_body_positions(fly.name)[thorax_idx], dtype=float)


def heading_vec() -> np.ndarray:
    q = np.asarray(sim.get_body_rotations(fly.name)[thorax_idx], dtype=float)
    fwd = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    return fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def on_key(keycode: int) -> None:
    global candy_active, feed_time, stick_t0, stick_from, stick_to
    p = thorax_pos()
    h = heading_vec()
    perp = np.array([-h[1], h[0], 0.0])
    if keycode in (67, 99):  # C
        offset = h * 12.0 + perp * rng.uniform(-4, 4)
        data.mocap_pos[candy_mid] = p + offset + np.array([0, 0, 0.7])
        candy_active = True
        feed_time = 0.0
        print(f"[candy] dropped at {np.round(data.mocap_pos[candy_mid][:2], 1)}")
    elif keycode in (83, 115):  # S
        side = 1.0 if rng.random() < 0.5 else -1.0
        stick_from = p - h * 14.0 + perp * side * 9.0 + np.array([0, 0, 1.5])
        stick_to = p - h * 1.0 + perp * side * 2.5 + np.array([0, 0, 1.5])
        stick_t0 = data.time
        print(f"[stick] strike from {'left' if side > 0 else 'right'}")


def apply_drive(mult: float, ds: np.ndarray, amp: float = 1.0) -> None:
    freqs = base_freqs * mult
    if ds[0] < 0:
        freqs[:3] *= -1
    if ds[1] < 0:
        freqs[3:] *= -1
    cpg_network.intrinsic_freqs = freqs
    cpg_network.intrinsic_amps = np.repeat(np.abs(ds)[:, None], 3, axis=1).ravel() * amp


print("Playground ready. Click the MuJoCo window: C=candy  S=stick  Esc=quit")

steps_per_frame = 10
with mj.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        t = data.time
        p = thorax_pos()
        h = heading_vec()
        mult, ds, amp = cruise_mult, np.array([1.0, 1.0]), 1.0

        # stick strike animation
        if stick_t0 is not None:
            s = (t - stick_t0) / 0.5
            if s >= 1.0:
                data.mocap_pos[stick_mid] = HIDDEN
                stick_t0 = None
            else:
                data.mocap_pos[stick_mid] = stick_from + (stick_to - stick_from) * min(s / 0.6, 1.0)
                if np.linalg.norm(data.mocap_pos[stick_mid][:2] - p[:2]) < 6.0:
                    rel = data.mocap_pos[stick_mid][:2] - p[:2]
                    cross_z = h[0] * rel[1] - h[1] * rel[0]
                    escape_ds = np.array([1.0, -1.0]) if cross_z > 0 else np.array([-1.0, 1.0])
                    escape_until = t + 0.9

        if t < escape_until:
            mult, ds = 1.7, escape_ds
        elif candy_active and np.linalg.norm(data.mocap_pos[candy_mid][:2] - p[:2]) < 5.5:
            mult, amp = 0.12, 0.5
            feed_time += sim.timestep
            if feed_time > 2.5:
                data.mocap_pos[candy_mid] = HIDDEN
                candy_active = False
        apply_drive(mult, ds, amp)

        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        viewer.sync()
