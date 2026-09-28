"""Live brain-body closed loop playground.

The GPU brain (metal_sim, 138k neurons) runs concurrently with the body.
Every 50ms of simulated time:
  body advances 50ms -> GPU brain advances 50ms -> P9 spikes read out
  -> P9 firing rate sets the CPG walking frequency (0.4x .. 1.6x of 12Hz).

Interactions:
  C - drop candy: when the fly gets close, sugar GRNs are REALLY stimulated
      at 200Hz in the GPU brain (whole-brain response computed live; the
      validated model keeps P9 silent -> fly slows to feed).
  S - stick strike: escape = turn away (CPG steering) + P9 neurons REALLY
      stimulated at 150Hz -> strong forward command -> sprint.

Keys (click the MuJoCo window first): C candy, S stick, Esc quit.
"""
import mujoco as mj
import numpy as np
from pathlib import Path
import sys

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

sys.path.insert(0, str(Path.home() / "fly-brain" / "metal_sim"))
from brain_bridge import MetalBrain  # noqa: E402

TICK_S = 0.05
TICK_BODY_STEPS = int(TICK_S / 1e-4)
TICK_BRAIN_STEPS = int(TICK_S * 1000 / 0.1)  # brain dt = 0.1ms
BASE_FREQ = 12.0
DRIVE_NORM = 105.0       # P9 Hz at full stimulation (measured on metal_sim)
FREQ_MIN_MULT = 0.4
FREQ_MAX_MULT = 1.6
rng = np.random.default_rng(7)

brain = MetalBrain()
print("GPU brain online:", brain.n, "neurons,", brain.n_syn, "synapses")

fly = make_locomotion_fly(name="live_fly", add_adhesion=True, colorize=True)
world = FlatGroundWorld()

# MjSpec quirk (flygym 2.1 / mujoco MjSpec): the first body added to the
# worldbody gets consumed by the attach+compile sequence, so feed it a
# sacrificial anchor; everything added after survives intact.
anchor = world.mjcf_root.worldbody.add_body(name="anchor", pos=[0, 0, -3])
anchor.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                rgba=[0, 0, 0, 0], contype=0, conaffinity=0)

candy_body = world.mjcf_root.worldbody.add_body(name="candy", pos=[0, 0, -50])
candy_body.mocap = True
candy_body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[1.2, 0, 0],
                    rgba=[1.0, 0.55, 0.1, 0.95], contype=0, conaffinity=0)
stick_body = world.mjcf_root.worldbody.add_body(name="stick", pos=[0, 0, -50])
stick_body.mocap = True
stick_body.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.5, 9, 0],
                    rgba=[0.45, 0.3, 0.15, 0.95], contype=0, conaffinity=0)

world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))

fly.add_tracking_camera(name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
                        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0)
sim = Simulation(world)
model, data = sim.mj_model, sim.mj_data
candy_mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "candy")]
stick_mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "stick")]
HIDDEN = np.array([0.0, 0.0, -50.0])

preprogrammed_steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")
cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep, intrinsic_amplitude=1.0,
    coupling_strength=10.0, convergence_coef=20.0, seed=0,
)
base_freqs = cpg_network.intrinsic_freqs.copy()
controller = CPGController(cpg_network=cpg_network,
                           preprogrammed_steps=preprogrammed_steps,
                           output_dof_order=dof_order)

sim.reset()
apply_locomotion_action(sim, fly.name, LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool)))
sim.warmup()

candy_active = False
feed_time = 0.0
sugar_stim_on = False
p9_stim_on = False
p9_stim_off_t = -1.0
stick_t0 = None
stick_from = stick_to = None
escape_ds = np.array([1.0, 1.0])
thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
p9_all = np.concatenate([brain.groups["p9_left"], brain.groups["p9_right"]])


def thorax_pos() -> np.ndarray:
    return np.asarray(sim.get_body_positions(fly.name)[thorax_idx], dtype=float)


def heading_vec() -> np.ndarray:
    q = np.asarray(sim.get_body_rotations(fly.name)[thorax_idx], dtype=float)
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    fwd = r @ np.array([1.0, 0.0, 0.0])
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    return fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def on_key(keycode: int) -> None:
    global candy_active, feed_time, sugar_stim_on, stick_t0, stick_from, stick_to
    global p9_stim_off_t
    p = thorax_pos()
    h = heading_vec()
    perp = np.array([-h[1], h[0], 0.0])
    if keycode in (67, 99):  # C
        data.mocap_pos[candy_mid] = p + h * 12.0 + perp * rng.uniform(-4, 4) + np.array([0, 0, 0.7])
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


def set_sugar(on: bool) -> None:
    global sugar_stim_on
    if on != sugar_stim_on:
        brain.set_rate("sugar", 200.0 if on else 0.0)
        sugar_stim_on = on


def set_p9(on: bool) -> None:
    global p9_stim_on
    if on != p9_stim_on:
        brain.set_rate("p9_left", 150.0 if on else 0.0)
        brain.set_rate("p9_right", 150.0 if on else 0.0)
        p9_stim_on = on


print("Live closed loop ready. Click MuJoCo window: C=candy S=stick Esc=quit")

with mj.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        # ---------- body-level events ----------
        t = data.time
        p = thorax_pos()
        h = heading_vec()
        ds = np.array([1.0, 1.0])
        amp = 1.0

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
                    set_p9(True)   # real P9 stimulation = the "run!" command
                    p9_stim_off_t = t + 0.9

        near_candy = candy_active and np.linalg.norm(
            data.mocap_pos[candy_mid][:2] - p[:2]) < 5.5
        if near_candy:
            set_sugar(True)    # real sugar-GRN stimulation = "food here"
            amp = 0.5
            feed_time += TICK_S
            if feed_time > 2.5:
                data.mocap_pos[candy_mid] = HIDDEN
                candy_active = False
                print("[candy] eaten")
        else:
            set_sugar(False)

        if p9_stim_on and t >= p9_stim_off_t:
            set_p9(False)      # escape over -> back to brain-idle walk

        # ---------- one closed-loop tick: brain -> readout -> body ----------
        brain.run(steps=TICK_BRAIN_STEPS)
        sp_t, sp_n = brain.drain_spikes()
        p9_hz = np.isin(sp_n, p9_all).sum() / 2.0 / TICK_S
        drive = min(p9_hz / DRIVE_NORM, 1.0)
        mult = FREQ_MIN_MULT + (FREQ_MAX_MULT - FREQ_MIN_MULT) * drive
        if near_candy:
            mult = min(mult, 0.25)   # feeding stance on top of brain drive
        apply_drive(mult, ds, amp)

        for _ in range(TICK_BODY_STEPS):
            action = controller.step()
            apply_locomotion_action(sim, fly.name, action)
            sim.step()
        viewer.sync()

set_sugar(False)
set_p9(False)
brain.close()
