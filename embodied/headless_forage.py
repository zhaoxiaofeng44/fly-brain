"""Headless full-loop verification of the nature playground foraging logic.

Same controller stack as interactive_playground_nature.py (GPU brain + CPG body
+ reflex steering), minus the MuJoCo viewer. Runs until all food is eaten or
the time limit hits, logging the same status lines.
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
TICK_BRAIN_STEPS = int(TICK_S * 1000 / 0.1)
DRIVE_NORM = 105.0
FREQ_MIN_MULT = 0.4
FREQ_MAX_MULT = 1.6
N_TREES = 6
N_ROCKS = 5
N_CANDIES = 6
ODOR_L = 25.0
EAT_DIST = 2.8
EAT_TIME = 2.0
FORAGE_HZ = 80.0
ESCAPE_HZ = 150.0
TIME_LIMIT_S = 150.0

scene_rng = np.random.default_rng(11)
brain = MetalBrain()
print("GPU brain online:", brain.n, "neurons,", brain.n_syn, "synapses", flush=True)

fly = make_locomotion_fly(name="live_fly", add_adhesion=True, colorize=True)
world = FlatGroundWorld()
anchor = world.mjcf_root.worldbody.add_body(name="anchor", pos=[0, 0, -3])
anchor.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                rgba=[0, 0, 0, 0], contype=0, conaffinity=0)

obstacles = []
for i in range(N_TREES):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(24, 38) if i < 4 else scene_rng.uniform(14, 22)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    trunk_r = scene_rng.uniform(1.6, 2.6)
    h = scene_rng.uniform(9, 14)
    body = world.mjcf_root.worldbody.add_body(name=f"tree{i}", pos=[x, y, 0])
    body.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[trunk_r, h / 2, 0],
                  pos=[0, 0, h / 2], rgba=[0.40, 0.26, 0.12, 1],
                  contype=1, conaffinity=1)
    body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[4.5, 0, 0],
                  pos=[0, 0, h + 2.0], rgba=[0.20, 0.45, 0.15, 1])
    obstacles.append((float(x), float(y), trunk_r + 1.5))
for i in range(N_ROCKS):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(10, 34)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    rx, ry, rz = (scene_rng.uniform(2.2, 4.5), scene_rng.uniform(1.8, 3.6),
                  scene_rng.uniform(1.2, 2.4))
    body = world.mjcf_root.worldbody.add_body(name=f"rock{i}", pos=[x, y, 0])
    body.add_geom(type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=[rx, ry, rz],
                  pos=[0, 0, rz * 0.45], rgba=[0.52, 0.51, 0.49, 1],
                  contype=1, conaffinity=1)
    obstacles.append((float(x), float(y), max(rx, ry) + 1.5))
for i in range(18):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(6, 38)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    hl = scene_rng.uniform(0.7, 1.3)
    body = world.mjcf_root.worldbody.add_body(name=f"grass{i}", pos=[x, y, 0])
    body.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.13, hl, 0],
                  pos=[0, 0, hl], rgba=[0.24, 0.55, 0.18, 1],
                  contype=0, conaffinity=0)
candy_bodies = []
for i in range(N_CANDIES):
    b = world.mjcf_root.worldbody.add_body(name=f"food{i}", pos=[0, 0, -50])
    b.mocap = True
    b.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[1.2, 0, 0],
               rgba=[1.0, 0.55, 0.1, 0.95], contype=0, conaffinity=0)
    candy_bodies.append(b)

world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
sim = Simulation(world)
model, data = sim.mj_model, sim.mj_data
candy_mids = [model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"food{i}")]
              for i in range(N_CANDIES)]
HIDDEN = np.array([0.0, 0.0, -50.0])


def food_visible(i):
    return data.mocap_pos[candy_mids[i]][2] > -40


def sample_food_xy(placed):
    for _ in range(200):
        a = scene_rng.uniform(0, 2 * np.pi)
        r = scene_rng.uniform(8, 30)
        x, y = r * np.cos(a), r * np.sin(a)
        if np.hypot(x, y) < 7.0:
            continue
        if any(np.hypot(x - ox, y - oy) < cr + 3.5 for ox, oy, cr in obstacles):
            continue
        if any(np.hypot(x - fx, y - fy) < 6.0 for fx, fy in placed):
            continue
        return np.array([x, y])
    return np.array([x, y])


placed = []
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

for i in range(N_CANDIES):
    xy = sample_food_xy(placed)
    placed.append(tuple(xy))
    data.mocap_pos[candy_mids[i]] = np.array([xy[0], xy[1], 0.7])
print(f"[scene] food at {[tuple(np.round(q, 1)) for q in placed]}", flush=True)

thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
p9_all = np.concatenate([brain.groups["p9_left"], brain.groups["p9_right"]])


def thorax_pos():
    return np.asarray(sim.get_body_positions(fly.name)[thorax_idx], dtype=float)


def heading_vec():
    q = np.asarray(sim.get_body_rotations(fly.name)[thorax_idx], dtype=float)
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    fwd = r @ np.array([1.0, 0.0, 0.0])
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    return fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def apply_drive(mult, ds, amp=1.0, amp_sides=None):
    freqs = base_freqs * mult
    if ds[0] < 0:
        freqs[:3] *= -1
    if ds[1] < 0:
        freqs[3:] *= -1
    cpg_network.intrinsic_freqs = freqs
    if amp_sides is None:
        amps = np.repeat(np.abs(ds)[:, None], 3, axis=1).ravel()
    else:
        amps = np.array([amp_sides[0]] * 3 + [amp_sides[1]] * 3)
    cpg_network.intrinsic_amps = amps * amp


def set_sugar(on):
    brain.set_rate("sugar", 200.0 if on else 0.0)


def set_p9_hz(hz):
    brain.set_rate("p9_left", hz)
    brain.set_rate("p9_right", hz)


feed_time = 0.0
eaten = 0
u_prev = 0.0
p9_smooth = 0.0
last_log = -1.0
last_pos = np.zeros(2)
still_t = 0.0
recover_t = 0.0
recover_dir = 1.0
last_tick_pos = None
ramp_t = 0.0
sugar_on = False

while data.time < TIME_LIMIT_S and eaten < N_CANDIES:
    t = data.time
    p = thorax_pos()
    h = heading_vec()
    ds = np.array([1.0, 1.0])
    amp = 1.0
    amp_sides = None

    vis = [i for i in range(N_CANDIES) if food_visible(i)]
    target_i, d, psi = -1, np.inf, 0.0
    if vis:
        xy = np.array([data.mocap_pos[candy_mids[i]][:2] for i in vis])
        dd = np.linalg.norm(xy - p[:2], axis=1)
        k = int(np.argmin(dd))
        target_i, d = vis[k], float(dd[k])
        rel = xy[k] - p[:2]
        psi = float(np.arctan2(h[0] * rel[1] - h[1] * rel[0],
                               h[0] * rel[0] + h[1] * rel[1]))

    feeding = target_i >= 0 and d < EAT_DIST
    if feeding:
        if not sugar_on:
            set_sugar(True)
            sugar_on = True
        amp = 0.5
        feed_time += TICK_S
        if feed_time > EAT_TIME:
            data.mocap_pos[candy_mids[target_i]] = HIDDEN
            eaten += 1
            print(f"[food] eaten ({eaten}/{N_CANDIES}) at t={t:.1f}s", flush=True)
            set_sugar(False)
            sugar_on = False
            feed_time = 0.0
            ramp_t = 0.6
            target_i, d = -1, np.inf
    else:
        if sugar_on:
            set_sugar(False)
            sugar_on = False
        feed_time = 0.0

    if feeding:
        set_p9_hz(0.0)
        state = "feed"
    elif target_i >= 0:
        set_p9_hz(FORAGE_HZ)
        state = "forage"
    else:
        set_p9_hz(0.0)
        state = "wander"

    u = 0.0
    spin = False
    if feeding:
        still_t = 0.0
    else:
        # stuck-detection: barely moving while drive is on -> spin recovery
        if last_tick_pos is not None:
            disp = float(np.linalg.norm(p[:2] - last_tick_pos))
            if disp < 0.08:
                still_t += TICK_S
            else:
                still_t = 0.0
        last_tick_pos = p[:2].copy()
        if recover_t > 0.0:
            recover_t -= TICK_S
            spin = True
            ds = np.array([-1.0, 1.0]) if recover_dir > 0 else np.array([1.0, -1.0])
        elif still_t > 2.0:
            recover_t = 0.8
            recover_dir = -recover_dir
            still_t = 0.0
            print(f"[recover] stuck -> spin burst at t={t:.1f}s", flush=True)
        if target_i >= 0:
            conc = float(np.exp(-d / ODOR_L))
            if abs(psi) > 1.2 and d > 2.5:
                spin = True
                ds = np.array([-1.0, 1.0]) if psi > 0 else np.array([1.0, -1.0])
            elif abs(psi) > 0.12:
                u = float(np.clip((0.35 + 1.2 * conc) * psi, -0.7, 0.7))
                u = 0.6 * u_prev + 0.4 * u
        else:
            u = 0.3 * np.sin(0.6 * t)
        if not spin:
            for ox, oy, cr in obstacles:
                relx, rely = ox - p[0], oy - p[1]
                do = float(np.hypot(relx, rely))
                if 1e-6 < do < cr + 6.0:
                    w = 0.9 * (1.0 - do / (cr + 6.0))
                    u -= w if h[0] * rely - h[1] * relx > 0 else -w
            u = float(np.clip(u, -0.7, 0.7))
            amp_sides = (1.0 - 0.55 * max(u, 0.0), 1.0 - 0.55 * max(-u, 0.0))
    u_prev = u

    brain.run(steps=TICK_BRAIN_STEPS)
    sp_t, sp_n = brain.drain_spikes()
    p9_hz = np.isin(sp_n, p9_all).sum() / 2.0 / TICK_S
    p9_smooth = 0.6 * p9_smooth + 0.4 * p9_hz
    drive = min(p9_smooth / DRIVE_NORM, 1.0)
    mult = FREQ_MIN_MULT + (FREQ_MAX_MULT - FREQ_MIN_MULT) * drive
    if feeding:
        mult = min(mult, 0.25)
    elif ramp_t > 0.0:
        mult = min(mult, 1.0)
        ramp_t -= TICK_S
    elif target_i >= 0 and d < 5.0 and abs(psi) < 0.9:
        mult *= 0.6
    apply_drive(mult, ds, amp, amp_sides)

    for _ in range(TICK_BODY_STEPS):
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step()

    if t - last_log > 5.0:
        spd = float(np.linalg.norm(p[:2] - last_pos)) / max(t - last_log, 1e-6)
        last_pos = p[:2].copy()
        last_log = t
        tgt = (f"food{target_i} d={d:5.1f}mm psi={psi:+.2f}"
               if target_i >= 0 else "no food")
        print(f"[{t:6.1f}s] {state:7s} {tgt} u={u:+.2f} "
              f"p9={p9_hz:5.1f}Hz mult={mult:.2f} spd={spd:5.1f}mm/s "
              f"pos=({p[0]:6.1f},{p[1]:6.1f},{p[2]:4.1f}) still={still_t:4.1f}s",
              flush=True)

print(f"RESULT: eaten={eaten}/{N_CANDIES} at t={data.time:.1f}s", flush=True)
set_sugar(False)
set_p9_hz(0.0)
brain.close()
