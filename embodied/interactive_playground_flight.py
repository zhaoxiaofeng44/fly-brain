"""Canopy playground: a brain-driven fly perches, descends to forage, flies
back up to rest, and repeats — behavioral timers are gone; every transition
is gated on the live brain's measured spike output.

Honest breakdown of what is real vs. surrogate:
  * Brain      - the 138k-neuron Metal simulation runs every tick; the body
    only acts on what its spike output actually says.
  * Decisions  - hunger is an internal model (the connectome has no annotated
    hunger circuitry); it is expressed INTO the brain as hunger-scaled P9
    stimulation. Descent fires when measured P9 arousal crosses TAKEOFF_P9
    (and the fly is hungry); after feeding the real sugar-GRN -> P9
    suppression satiety circuit silences the readout, and that measured
    silence sends the fly back up to rest.
  * Speed      - measured P9 rate -> CPG frequency (real pathway).
  * Steering   - an odor-gradient reflex computes turn intent; the intent is
    sent to the brain as differential left/right P9 stimulation, and the CPG
    turn command is derived from the measured L/R spike differential. Hard
    spins / escape stay direct reflexes.
  * Flight     - SCRIPTED surrogate (no flight circuitry in the connectome,
    welded decorative wings): gravity-compensated bezier guidance +
    kinematically held attitude + kinematic wingbeat. Whether/when it flies
    is brain-driven; how it flies is not.
  * Escape / righting - hard reflexes (stick threat, belly-flat hop).

Keys (click the MuJoCo window first): C drop candy, S stick strike,
F force a flight (manual override), Esc quit.
"""
import os
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
EAT_DIST = 2.8          # enter feeding range
EAT_HOLD = 4.0          # hysteresis: once feeding, stay until this far
EAT_TIME = 1.2
ESCAPE_HZ = 150.0

# --- brain-gated motivation (no behavioral timers) ---
P9_STIM_MIN = 8.0       # motivation stimulation at hunger 0 (Hz)
P9_STIM_MAX = 95.0      # motivation stimulation at hunger 1 (Hz)
HUNGER_T = 30.0         # satiety -> starving over this many seconds
TAKEOFF_P9 = 47.0       # measured P9 arousal (Hz) that commands a descent
HUNGER_TAKEOFF = 0.40   # arousal alone is not enough: must be hungry too
REST_P9 = 23.0          # measured P9 below this (satiety) -> fly up to rest
MIN_STATE_T = 1.5       # dwell time before a brain-gated transition
SEARCH_GIVEUP = 25.0    # wandering this long without food -> relocate
STEER_GAIN = 0.8        # depth of turn intent on the L/R P9 differential
FLIGHT_HZ = 80.0        # P9 drive expressed while airborne
FLY_SPEED = 22.0        # cruise speed of the guided arc (units/s)
KP = 120.0              # guidance stiffness (1/s^2), physics-rate PD
KD = 25.0               # guidance damping (1/s)
WING_AMP = 0.9          # wingbeat amplitude (rad)
WING_HZ = 7.0           # wingbeat frequency (visual)

scene_rng = np.random.default_rng(11)

brain = MetalBrain()
print("GPU brain online:", brain.n, "neurons,", brain.n_syn, "synapses", flush=True)

fly = make_locomotion_fly(name="live_fly", add_adhesion=True, colorize=True)

# Wing hinges (no actuators; driven kinematically during flight, held at 0
# otherwise). flygym's actuator/joint bookkeeping only covers leg DoFs, so
# these are invisible to it.
wing_seg_cls = type(fly).BODY_SEGMENT_CLASS
for side in ("l", "r"):
    wbody = fly.bodyseg_to_mjcfbody[wing_seg_cls(f"{side}_wing")]
    wbody.add_joint(type=mj.mjtJoint.mjJNT_HINGE, name=f"{side}_wing_flap",
                    axis=[1, 0, 0])

world = FlatGroundWorld()

# MjSpec quirk: the first body added to the worldbody gets consumed by the
# attach+compile sequence, so feed it a sacrificial anchor.
anchor = world.mjcf_root.worldbody.add_body(name="anchor", pos=[0, 0, -3])
anchor.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0],
                rgba=[0, 0, 0, 0], contype=0, conaffinity=0)

obstacles = []       # (x, y, clear_radius) for ground steering
canopy_tops = []     # (x, y, z) landing spots on each tree
scene_geoms = []     # named geoms the fly can physically land on / bump into

for i in range(N_TREES):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(24, 38) if i < 4 else scene_rng.uniform(14, 22)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    trunk_r = scene_rng.uniform(1.6, 2.6)
    h = scene_rng.uniform(9, 14)
    canopy_r = 4.2 + scene_rng.uniform(0.0, 0.8)
    tint = scene_rng.uniform(0.0, 0.08)
    body = world.mjcf_root.worldbody.add_body(name=f"tree{i}", pos=[x, y, 0])
    g = body.add_geom(name=f"tree{i}_trunk", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=[trunk_r, h / 2, 0], pos=[0, 0, h / 2],
                      rgba=[0.40, 0.26, 0.12, 1], contype=1, conaffinity=1)
    scene_geoms.append(g)
    g = body.add_geom(name=f"tree{i}_canopy", type=mj.mjtGeom.mjGEOM_SPHERE,
                      size=[canopy_r, 0, 0], pos=[0, 0, h + 2.0],
                      rgba=[0.18, 0.42 + tint, 0.14, 1])
    scene_geoms.append(g)
    body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[2.8, 0, 0],
                  pos=[1.8, -1.2, h + 1.0], rgba=[0.22, 0.48 + tint, 0.16, 1])
    # flat perch pad on the canopy apex — a dome is too slick to stand on
    g = body.add_geom(name=f"tree{i}_pad", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=[3.2, 0.18, 0], pos=[0, 0, h + 2.0 + canopy_r],
                      rgba=[0.30, 0.36, 0.18, 1], contype=1, conaffinity=1,
                      friction=[3.0, 0.005, 0.0001])
    scene_geoms.append(g)
    obstacles.append((float(x), float(y), trunk_r + 1.5))
    canopy_tops.append((float(x), float(y), h + 2.0 + canopy_r + 0.18))

for i in range(N_ROCKS):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(10, 34)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    rx, ry, rz = (scene_rng.uniform(2.2, 4.5), scene_rng.uniform(1.8, 3.6),
                  scene_rng.uniform(1.2, 2.4))
    sh = scene_rng.uniform(-0.04, 0.04)
    body = world.mjcf_root.worldbody.add_body(name=f"rock{i}", pos=[x, y, 0])
    geom = body.add_geom(name=f"rock{i}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                         size=[rx, ry, rz], pos=[0, 0, rz * 0.45],
                         rgba=[0.52 + sh, 0.51 + sh, 0.49 + sh, 1],
                         contype=1, conaffinity=1)
    yaw = scene_rng.uniform(0, np.pi)
    geom.quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
    scene_geoms.append(geom)
    obstacles.append((float(x), float(y), max(rx, ry) + 1.5))

for i in range(18):
    ang = scene_rng.uniform(0, 2 * np.pi)
    rad = scene_rng.uniform(6, 38)
    x, y = rad * np.cos(ang), rad * np.sin(ang)
    hl = scene_rng.uniform(0.7, 1.3)
    body = world.mjcf_root.worldbody.add_body(name=f"grass{i}", pos=[x, y, 0])
    geom = body.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE,
                         size=[0.13, hl, 0], pos=[0, 0, hl],
                         rgba=[0.24, 0.52 + scene_rng.uniform(0, 0.1), 0.18, 1],
                         contype=0, conaffinity=0)
    a = scene_rng.uniform(0, 0.35)
    n = scene_rng.uniform(0, 2 * np.pi)
    geom.quat = [float(np.cos(a / 2)), float(np.cos(n) * np.sin(a / 2)),
                 float(np.sin(n) * np.sin(a / 2)), 0.0]

candy_bodies = []
for i in range(N_CANDIES):
    b = world.mjcf_root.worldbody.add_body(name=f"food{i}", pos=[0, 0, -50])
    b.mocap = True
    b.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[1.2, 0, 0],
               rgba=[1.0, 0.55, 0.1, 0.95], contype=0, conaffinity=0)
    candy_bodies.append(b)

stick_body = world.mjcf_root.worldbody.add_body(name="stick", pos=[0, 0, -50])
stick_body.mocap = True
stick_body.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.5, 9, 0],
                    rgba=[0.45, 0.3, 0.15, 0.95], contype=0, conaffinity=0)

# the fly starts perched on tree 0's canopy (spawn just above the pad)
t0 = canopy_tops[0]
world.add_fly(fly, [t0[0], t0[1], t0[2] + 1.15], Rotation3D("quat", [1, 0, 0, 0]))

# the fly's collision geoms only touch via explicit pairs (flygym ground
# pattern) — register them against the scenery so perching/bumping is real
from flygym.anatomy import ContactBodiesPreset  # noqa: E402
_contact_segs = fly.CONTACT_BODIES_PRESET_CLASS.LEGS_THORAX_ABDOMEN_HEAD \
    .to_body_segments_list() if hasattr(fly, "CONTACT_BODIES_PRESET_CLASS") else []
if not _contact_segs:
    _contact_segs = [s for s in fly.get_bodysegs_order()
                     if s.is_leg() or s.link in ("c_thorax", "c_abdomen6")]
for _seg in _contact_segs:
    for _fg in fly.bodyseg_to_mjcfgeom.get(_seg, []):
        for _sg in scene_geoms:
            world.mjcf_root.add_pair(
                geomname1=_fg.name, geomname2=_sg.name,
                name=f"{_fg.name}-{_sg.name}-contact")

fly.add_tracking_camera(name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
                        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0)
sim = Simulation(world)
model, data = sim.mj_model, sim.mj_data
candy_mids = [model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"food{i}")]
              for i in range(N_CANDIES)]
stick_mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "stick")]
HIDDEN = np.array([0.0, 0.0, -50.0])
thorax_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "live_fly/c_thorax")
# free joint of the fly root — its dof address is NOT necessarily 0 because
# scenery bodies precede the fly in the compiled model
FREEJ_DOFADR = int(model.jnt_dofadr[
    mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "live_fly")])
wing_qadr = {}
for side in ("l", "r"):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{side}_wing_flap")
    wing_qadr[side] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])
# wings are visual-only: kill their mass/inertia so the kinematic wingbeat
# injects no reaction forces into the body
for side in ("l", "r"):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"live_fly/{side}_wing")
    model.body_mass[bid] = 1e-12
    model.body_inertia[bid] = [1e-12, 1e-12, 1e-12]
print(f"[model] mass={model.body_mass.sum():.3e} gravity={model.opt.gravity}",
      flush=True)


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
print(f"[scene] {N_TREES} trees (perch z "
      f"{[round(q[2], 1) for q in canopy_tops]}), {N_ROCKS} rocks, "
      f"{N_CANDIES} food at {[tuple(np.round(q, 1)) for q in placed]}",
      flush=True)
print(f"[start] perched on tree0 at z={canopy_tops[0][2]:.1f}", flush=True)

thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
p9_l_ids = brain.groups["p9_left"]
p9_r_ids = brain.groups["p9_right"]
GRAV = float(np.linalg.norm(model.opt.gravity))
# mass of the FLY only — the scene (trees/rocks) dominates body_mass.sum()
_fly_body_ids = [i for i in range(model.nbody)
                 if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or ""
                     ).startswith(fly.name + "/")]
MASS = float(sum(model.body_mass[i] for i in _fly_body_ids))
_im = model.body_mass[_fly_body_ids]     # for whole-fly CoM queries
print(f"[model] fly mass={MASS:.3e} gravity={GRAV:.1f}", flush=True)


def thorax_pos():
    return np.asarray(sim.get_body_positions(fly.name)[thorax_idx], dtype=float)


def thorax_vel():
    fa = FREEJ_DOFADR
    return np.asarray(data.qvel[fa:fa + 3], dtype=float)


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


def set_p9_lr(hz_l, hz_r):
    brain.set_rate("p9_left", max(hz_l, 0.0))
    brain.set_rate("p9_right", max(hz_r, 0.0))


ATT_RATE = 6.0            # attitude hold: proportional rate gain (1/s)
ATT_WMAX = 8.0            # attitude hold: max angular speed (rad/s)


def _quat_mul(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def stabilize_attitude(flight, dt):
    """Kinematic attitude hold for the scripted flight surrogate: integrate
    the freejoint quaternion toward the takeoff attitude (yawed to face the
    direction of travel). Torque-based PD cannot work here — the guidance
    force acts at the thorax CoM and the r x F moment (~3.6 in model units)
    dwarfs the fly's rotational inertia (~2.6e-3), so the body tumbles at
    ~140 rad/s before any reasonable PD can react."""
    v = thorax_vel()
    if float(np.hypot(v[0], v[1])) > 5.0:
        yaw = float(np.arctan2(v[1], v[0]))
    else:
        fw = heading_vec()
        yaw = float(np.arctan2(fw[1], fw[0]))
    d_yaw = float(np.angle(np.exp(1j * (yaw - flight.yaw0))))
    qz = np.array([np.cos(d_yaw / 2), 0.0, 0.0, np.sin(d_yaw / 2)])
    qd = _quat_mul(qz, flight.q0)
    qd /= np.linalg.norm(qd)
    fa = FREEJ_DOFADR
    qc = np.asarray(data.qpos[fa + 3:fa + 7], dtype=float)
    qe = _quat_mul(qd, np.array([qc[0], -qc[1], -qc[2], -qc[3]]))
    if qe[0] < 0.0:
        qe = -qe
    ang = 2.0 * float(np.arccos(min(1.0, qe[0])))
    rate_vec = np.zeros(3)
    if ang > 1e-6:
        s = float(np.sqrt(max(1e-12, 1.0 - qe[0] * qe[0])))
        rate_vec = qe[1:4] / s * min(ang * ATT_RATE, ATT_WMAX)
    dom = rate_vec * dt
    dn = float(np.linalg.norm(dom))
    if dn > 1e-12:
        dq = np.concatenate([[float(np.cos(dn / 2))],
                             dom / dn * float(np.sin(dn / 2))])
        qn = _quat_mul(dq, qc)
        data.qpos[fa + 3:fa + 7] = qn / np.linalg.norm(qn)
    w, x, y, z = qc                       # feed the integrator a matching
    r = np.array([                        # angular velocity (body-local)
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    data.qvel[fa + 3:fa + 6] = r.T @ rate_vec


class Flight:
    def __init__(self, target, label, goal="canopy"):
        self.p0 = thorax_pos().copy()
        self.q0 = np.asarray(data.xquat[thorax_body_id], dtype=float).copy()
        fw = heading_vec()
        self.yaw0 = float(np.arctan2(fw[1], fw[0]))
        self.p1 = np.asarray(target, dtype=float)
        self.label = label
        self.goal = goal
        self.t = 0.0
        mid = 0.5 * (self.p0 + self.p1)
        mid[2] = max(self.p0[2], self.p1[2]) + 6.0
        self.cp = mid
        dist = float(np.linalg.norm(self.p1 - self.p0))
        self.duration = float(np.clip(dist / FLY_SPEED, 1.5, 6.0))

    def body_force(self, dt):
        """Recompute guidance at physics rate; advance internal clock."""
        self.t += dt
        u = min(self.t / self.duration, 1.0)
        s = u * u * (3.0 - 2.0 * u)          # smoothstep easing
        ds = 6.0 * u * (1.0 - u) / self.duration
        p_des = ((1 - s) ** 2) * self.p0 + 2 * (1 - s) * s * self.cp + (s ** 2) * self.p1
        v_des = ds * (2 * (1 - s) * (self.cp - self.p0) + 2 * s * (self.p1 - self.cp))
        p = thorax_pos()
        v = thorax_vel()
        if self.goal == "ground" and p[2] < 3.0:
            p_des[0] = p[0]                  # flare: kill horizontal pull,
            p_des[1] = p[1]                  # descend vertically to the ground
        return MASS * (KP * (p_des - p) + KD * (v_des - v)) \
            + MASS * GRAV * np.array([0.0, 0.0, 1.0])

    def progress(self):
        u = min(self.t / self.duration, 1.0)
        dist = float(np.linalg.norm(self.p1 - thorax_pos()))
        return u, dist

    def done(self, u, dist, goal=None, z=None):
        if goal == "ground" and u > 0.5 and z is not None and z < 2.2:
            return True           # low enough over the ground: cut and drop
        return (u > 0.85 and dist < 1.0) or self.t > self.duration + 2.5


def set_wings(phase, flying):
    if flying:
        q = WING_AMP * np.sin(2 * np.pi * WING_HZ * phase)
        qd = WING_AMP * 2 * np.pi * WING_HZ * np.cos(2 * np.pi * WING_HZ * phase)
    else:
        q = qd = 0.0
    ql, qvl = wing_qadr["l"]
    qr, qvr = wing_qadr["r"]
    data.qpos[ql] = q
    data.qvel[qvl] = qd
    data.qpos[qr] = -q
    data.qvel[qvr] = -qd


state = "perch"          # perch | flight | walk
perch_i = 0
last_tree = 0
state_t = 0.0
feed_time = 0.0
feed_active = False
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
p9_cur = -1.0
flight = None
flight_goal = None       # "canopy" | "ground"
wing_phase = 0.0
stick_t0 = None
stick_from = stick_to = None
escape_ds = np.array([1.0, 1.0])
p9_stim_off_t = -1.0
last_meal = -18.0       # start at hunger 0.6: aroused, ready to descend
search_t = 0.0
p9_l_s = 0.0            # smoothed measured L/R P9 rates (Hz)
p9_r_s = 0.0
u_meas = 0.0
state2 = "perch"


def choose_tree(exclude):
    others = [i for i in range(N_TREES) if i != exclude]
    return int(scene_rng.choice(others))


def _takeoff_hop(flight):
    """Real flies jump before flying — also breaks contact wedges."""
    dir3 = np.zeros(3)
    dir_xy = flight.p1[:2] - flight.p0[:2]
    n = float(np.linalg.norm(dir_xy))
    if n > 1e-6:
        dir3[:2] = dir_xy / n * 18.0
    dir3[2] = 32.0
    fa = FREEJ_DOFADR
    data.qvel[fa:fa + 3] = dir3


def begin_flight_to_tree():
    global flight, flight_goal, state, state_t, last_tree
    tgt_i = choose_tree(last_tree)
    c = canopy_tops[tgt_i]
    flight = Flight([c[0], c[1], c[2] + 1.15], f"tree{last_tree}->tree{tgt_i}",
                    goal="canopy")
    flight_goal = "canopy"
    last_tree = tgt_i
    state = "flight"
    state_t = 0.0
    _takeoff_hop(flight)
    print(f"[flight] takeoff: {flight.label} (arc {flight.duration:.2f}s) "
          f"from=({flight.p0[0]:.1f},{flight.p0[1]:.1f},{flight.p0[2]:.1f}) "
          f"to=({flight.p1[0]:.1f},{flight.p1[1]:.1f},{flight.p1[2]:.1f}) "
          f"t={data.time:.2f}", flush=True)


def begin_flight_to_ground():
    global flight, flight_goal, state, state_t
    c = canopy_tops[last_tree]
    vis = [i for i in range(N_CANDIES) if food_visible(i)]
    bias = None
    if vis:                                 # scent-guided: aim near food
        xy = np.array([data.mocap_pos[candy_mids[i]][:2] for i in vis])
        k = int(np.argmin(np.linalg.norm(xy - c[:2], axis=1)))
        bias = xy[k]
    gx, gy = c[0], c[1]
    for _ in range(60):                     # pick a spot clear of obstacles
        ang = scene_rng.uniform(0, 2 * np.pi)
        if bias is not None:
            r = scene_rng.uniform(0.0, 5.0)
            gx, gy = bias[0] + r * np.cos(ang), bias[1] + r * np.sin(ang)
        else:
            r = scene_rng.uniform(5.0, 9.0)
            gx, gy = c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)
        if all(np.hypot(gx - ox, gy - oy) >= cr + 4.0
               for ox, oy, cr in obstacles):
            break
    flight = Flight([gx, gy, 1.2], f"tree{last_tree}->ground", goal="ground")
    flight_goal = "ground"
    state = "flight"
    state_t = 0.0
    _takeoff_hop(flight)
    print(f"[flight] descending to ground at ({gx:.1f},{gy:.1f})", flush=True)


def on_key(keycode: int) -> None:
    global stick_t0, stick_from, stick_to
    if keycode in (67, 99):  # C
        for i in range(N_CANDIES):
            if not food_visible(i):
                p, h = thorax_pos(), heading_vec()
                perp = np.array([-h[1], h[0], 0.0])
                spot = p + h * 12.0 + perp * scene_rng.uniform(-4, 4)
                data.mocap_pos[candy_mids[i]] = np.array([spot[0], spot[1], 0.7])
                print(f"[candy] dropped at {np.round(spot[:2], 1)}", flush=True)
                return
        print(f"[candy] pool empty ({N_CANDIES}/{N_CANDIES} on field)", flush=True)
    elif keycode in (83, 115):  # S: stick strike (matters while walking)
        p, h = thorax_pos(), heading_vec()
        perp = np.array([-h[1], h[0], 0.0])
        side = 1.0 if scene_rng.random() < 0.5 else -1.0
        stick_from = p - h * 14.0 + perp * side * 9.0 + np.array([0, 0, 1.5])
        stick_to = p - h * 1.0 + perp * side * 2.5 + np.array([0, 0, 1.5])
        stick_t0 = data.time
        print(f"[stick] strike from {'left' if side > 0 else 'right'}", flush=True)
    elif keycode in (70, 102):  # F: force a flight
        global flight, flight_goal, state, state_t
        if state == "perch":
            begin_flight_to_tree()
        elif state == "walk":
            begin_flight_to_tree()


print("Canopy playground ready. Click MuJoCo window: C=candy S=stick "
      "F=fly-to-tree Esc=quit", flush=True)

if os.environ.get("HEADLESS") == "1":
    limit = float(os.environ.get("SIM_LIMIT", "60"))

    class _DummyViewer:
        def is_running(self):
            return data.time < limit

        def sync(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    viewer_cm = _DummyViewer()
else:
    viewer_cm = mj.viewer.launch_passive(model, data, key_callback=on_key)

with viewer_cm as viewer:
    while viewer.is_running():
        t = data.time
        p = thorax_pos()
        h = heading_vec()
        ds = np.array([1.0, 1.0])
        amp = 1.0
        amp_sides = None
        flying = False
        state_t += TICK_S

        # ---------- stick strike (only while walking) ----------
        if stick_t0 is not None:
            s = (t - stick_t0) / 0.5
            if s >= 1.0:
                data.mocap_pos[stick_mid] = HIDDEN
                stick_t0 = None
            elif state == "walk":
                data.mocap_pos[stick_mid] = stick_from + (stick_to - stick_from) * min(s / 0.6, 1.0)
                if np.linalg.norm(data.mocap_pos[stick_mid][:2] - p[:2]) < 6.0:
                    cross_z = h[0] * (data.mocap_pos[stick_mid][1] - p[1]) \
                        - h[1] * (data.mocap_pos[stick_mid][0] - p[0])
                    escape_ds = np.array([1.0, -1.0]) if cross_z > 0 else np.array([-1.0, 1.0])
                    p9_stim_off_t = t + 0.9
        escaping = state == "walk" and t < p9_stim_off_t

        # ---------- food (only meaningful while walking) ----------
        vis = [i for i in range(N_CANDIES) if food_visible(i)]
        target_i, d, psi = -1, np.inf, 0.0
        if vis and state == "walk":
            xy = np.array([data.mocap_pos[candy_mids[i]][:2] for i in vis])
            dd = np.linalg.norm(xy - p[:2], axis=1)
            k = int(np.argmin(dd))
            target_i, d = vis[k], float(dd[k])
            rel = xy[k] - p[:2]
            psi = float(np.arctan2(h[0] * rel[1] - h[1] * rel[0],
                                   h[0] * rel[0] + h[1] * rel[1]))

        feeding = (not escaping) and target_i >= 0 and \
            (feed_active or d < EAT_DIST) and d < EAT_HOLD
        if feeding:
            feed_active = True
            if not sugar_on:
                set_sugar(True)
                sugar_on = True
            amp = 0.0            # stand still on the food while eating
            feed_time += TICK_S
            if feed_time > EAT_TIME:
                data.mocap_pos[candy_mids[target_i]] = HIDDEN
                eaten += 1
                last_meal = t       # hunger model resets; satiety itself is
                                    # the real sugar-GRN -> P9 suppression
                print(f"[food] eaten #{eaten} at t={t:.1f}s -> sugar-GRN "
                      f"satiety, hunger -> 0", flush=True)
                set_sugar(False)
                sugar_on = False
                feed_time = 0.0
                feed_active = False
                ramp_t = 0.6
                target_i, d = -1, np.inf
        else:
            feed_active = False
            if sugar_on:
                set_sugar(False)
                sugar_on = False
            feed_time = 0.0

        # ---------- state machine (gated on measured brain output) -------
        hunger = min(1.0, max(0.0, (t - last_meal) / HUNGER_T))
        if state == "perch":
            if os.environ.get("FLYDBG") == "1" and state_t < 1.0:
                v = thorax_vel()
                _w, _x, _y, _z = data.xquat[thorax_body_id]
                _upz = 1 - 2 * (_x * _x + _y * _y)
                print("[perchdbg] t=%.2f st=%.2f tilt=%.0f pos=(%.2f,%.2f,%.2f) "
                      "v=(%.1f,%.1f,%.1f) ncon=%d"
                      % (t, state_t,
                         float(np.degrees(np.arccos(np.clip(_upz, -1, 1)))),
                         p[0], p[1], p[2], v[0], v[1], v[2],
                         data.ncon), flush=True)
            if p[2] < canopy_tops[perch_i][2] - 3.0:
                state = "walk"       # fell off the canopy: forage instead
                state_t = 0.0
                print("[perch] fell off -> walking", flush=True)
            elif state_t > MIN_STATE_T and p9_smooth > TAKEOFF_P9 \
                    and hunger > HUNGER_TAKEOFF:
                print(f"[brain] AROUSAL p9={p9_smooth:.0f}Hz>"
                      f"{TAKEOFF_P9:.0f} hunger={hunger:.2f} "
                      f"-> descend & forage", flush=True)
                begin_flight_to_ground()
        elif state == "walk":
            if target_i >= 0 or feeding:
                search_t = 0.0
            else:
                search_t += TICK_S
            if not feeding and not escaping and state_t > MIN_STATE_T \
                    and p9_smooth < REST_P9:
                print(f"[brain] SATIETY p9={p9_smooth:.0f}Hz<"
                      f"{REST_P9:.0f} hunger={hunger:.2f} "
                      f"-> fly up to rest", flush=True)
                begin_flight_to_tree()
            elif still_t > 6.0:
                still_t = 0.0        # wedged: relocate by air
                begin_flight_to_ground()
            elif search_t > SEARCH_GIVEUP:
                search_t = 0.0       # nothing found here: try elsewhere
                begin_flight_to_ground()

        # ---------- flight guidance (forces applied at physics rate) ------
        if state == "flight" and flight is not None:
            flying = True
            amp = 0.0
        else:
            flying = False
            data.xfrc_applied[thorax_body_id, :] = 0

        # ---------- motivation: hunger -> P9 stimulation; the body then
        # acts on the brain's measured output, not on hunger itself ------
        base_stim = P9_STIM_MIN + (P9_STIM_MAX - P9_STIM_MIN) * hunger
        if escaping:
            set_p9_lr(ESCAPE_HZ, ESCAPE_HZ)
            state2 = "escape"
        elif feeding:
            set_p9_lr(0.0, 0.0)      # sugar-GRN satiety input, no drive
            state2 = "feed"
        elif state == "flight":
            set_p9_lr(FLIGHT_HZ, FLIGHT_HZ)
            state2 = "flight"
        elif state == "perch":
            set_p9_lr(base_stim, base_stim)   # arousal builds while resting
            state2 = "perch"
        else:
            # turn intent (from last tick's reflex) as an L/R rate diff
            stim_l = base_stim * (1.0 - STEER_GAIN * u)
            stim_r = base_stim * (1.0 + STEER_GAIN * u)
            set_p9_lr(stim_l, stim_r)
            state2 = "forage" if target_i >= 0 else "wander"

        # ---------- steering (reflex layer, walking only) ----------
        u = 0.0
        spin = False
        if state == "flight":
            pass
        elif escaping:
            ds = escape_ds.copy()
            still_t = 0.0
        elif feeding:
            still_t = 0.0
        elif state == "perch":
            ds = np.array([1.0, 1.0])
            amp = 0.0            # stance only; adhesion pins all feet
            u_prev = 0.0
        else:
            if last_tick_pos is not None:   # stuck-detection -> spin recovery
                disp = float(np.linalg.norm(p[:2] - last_tick_pos))
                still_t = still_t + TICK_S if disp < 0.08 else 0.0
            last_tick_pos = p[:2].copy()
            if p[2] < 0.7 and still_t > 1.5:   # belly-flat: righting hop
                fa = FREEJ_DOFADR
                data.qvel[fa:fa + 3] = [scene_rng.uniform(-30, 30),
                                        scene_rng.uniform(-30, 30), 350.0]
                data.qvel[fa + 3:fa + 6] = [scene_rng.uniform(-40, 40),
                                            scene_rng.uniform(-40, 40), 0.0]
                still_t = 0.0
                print("[righting] pop-up hop", flush=True)
            elif recover_t > 0.0:
                recover_t -= TICK_S
                spin = True
                ds = np.array([-1.0, 1.0]) if recover_dir > 0 else np.array([1.0, -1.0])
            elif still_t > 2.0:
                recover_t = 0.8
                recover_dir = -recover_dir
                still_t = 0.0
                print("[recover] stuck -> spin burst", flush=True)
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
                rb = float(np.hypot(p[0], p[1]))
                if rb > 40.0:
                    u += 1.0 if h[0] * (-p[1]) - h[1] * (-p[0]) > 0 else -1.0
                u = float(np.clip(u, -0.7, 0.7))
        u_prev = u

        # ---------- one closed-loop tick: brain -> readout -> body ----------
        brain.run(steps=TICK_BRAIN_STEPS)
        sp_t, sp_n = brain.drain_spikes()
        n_l = float(np.isin(sp_n, p9_l_ids).sum()) / TICK_S
        n_r = float(np.isin(sp_n, p9_r_ids).sum()) / TICK_S
        p9_l_s = 0.6 * p9_l_s + 0.4 * n_l
        p9_r_s = 0.6 * p9_r_s + 0.4 * n_r
        p9_hz = 0.5 * (n_l + n_r)
        p9_smooth = 0.6 * p9_smooth + 0.4 * p9_hz
        drive = min(p9_smooth / DRIVE_NORM, 1.0)
        mult = FREQ_MIN_MULT + (FREQ_MAX_MULT - FREQ_MIN_MULT) * drive
        if feeding:
            mult = min(mult, 0.25)
        elif ramp_t > 0.0:
            mult = min(mult, 1.0)
            ramp_t -= TICK_S
        elif target_i >= 0 and d < 5.0 and not escaping and abs(psi) < 0.9 \
                and state == "walk":
            mult *= 0.6
        if state == "perch":
            mult = 0.3
        if state == "walk" and not escaping and not feeding and not spin:
            # the CPG turn command comes from the brain's measured L/R
            # spike differential, not directly from the reflex
            tot = p9_l_s + p9_r_s
            raw = (p9_r_s - p9_l_s) / max(tot, 1e-6) * 2.0
            u_meas = float(np.clip(raw, -1.0, 1.0)) if abs(raw) > 0.12 else 0.0
            amp_sides = (1.0 - 0.55 * max(u_meas, 0.0),
                         1.0 - 0.55 * max(-u_meas, 0.0))
        apply_drive(mult, ds, amp, amp_sides)

        for _ in range(TICK_BODY_STEPS):
            if flying and flight is not None:
                f = flight.body_force(sim.timestep)
                data.xfrc_applied[thorax_body_id, :3] = f
                # cancel the r x F moment so the force pulls on the whole
                # fly CoM, not just the thorax (keeps linear dynamics clean)
                com = np.average(data.xipos[_fly_body_ids], axis=0,
                                 weights=_im)
                r = data.xipos[thorax_body_id] - com
                data.xfrc_applied[thorax_body_id, 3:6] = -np.cross(r, f)
                stabilize_attitude(flight, sim.timestep)
                set_wings(flight.t, True)
            else:
                set_wings(0.0, False)
            action = controller.step()
            if state == "perch":
                action.adhesion_onoff[:] = True   # grip, never release
            elif flying:
                action.adhesion_onoff[:] = False  # unstick tarsi in the air
            apply_locomotion_action(sim, fly.name, action)
            sim.step()
        set_wings(0.0, False)

        # ---------- flight completion ----------
        if flying and flight is not None:
            u, dist = flight.progress()
            if os.environ.get("FLYDBG") == "1":
                _w, _x, _y, _z = data.xquat[thorax_body_id]
                _upz = 1 - 2 * (_x * _x + _y * _y)   # body up-axis, world z
                tilt = float(np.degrees(np.arccos(np.clip(_upz, -1.0, 1.0))))
                print("[flydbg] t=%.2f ft=%.2f u=%.2f dist=%.1f tilt=%.0f "
                      "pos=(%.1f,%.1f,%.1f) warn=%d"
                      % (data.time, flight.t, u, dist, tilt,
                         p[0], p[1], p[2], data.warning[2].number), flush=True)
            if flight.done(u, dist, flight_goal, p[2]):
                data.xfrc_applied[thorax_body_id, :] = 0
                if flight_goal == "canopy":
                    state = "perch"
                    perch_i = last_tree
                    state_t = 0.0
                    print(f"[flight] landed on tree{last_tree} "
                          f"(t={data.time:.2f}s, dist={dist:.2f} "
                          f"ft={flight.t:.2f} u={u:.2f})", flush=True)
                else:
                    state = "walk"
                    state_t = 0.0
                    print(f"[flight] on the ground, foraging", flush=True)
                flight = None

        viewer.sync()

        if t - last_log > 2.0:
            spd = float(np.linalg.norm(p[:2] - last_pos)) / max(t - last_log, 1e-6)
            last_pos = p[:2].copy()
            last_log = t
            tgt = (f"food{target_i} d={d:5.1f}mm psi={psi:+.2f}"
                   if target_i >= 0 else "no food")
            print(f"[{t:6.1f}s] {state:7s} {state2:7s} h={hunger:.2f} {tgt} "
                  f"u={u:+.2f} p9={p9_hz:5.1f}Hz L{p9_l_s:4.0f}/R{p9_r_s:4.0f} "
                  f"mult={mult:.2f} spd={spd:5.1f}mm/s "
                  f"pos=({p[0]:6.1f},{p[1]:6.1f},{p[2]:5.1f}) still={still_t:4.1f}s",
                  flush=True)
            if spd < 0.3 and state == "walk":
                names = []
                for ci in range(min(data.ncon, 4)):
                    con = data.contact[ci]
                    names.append(
                        f"{mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, con.geom1)}"
                        f"<->{mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, con.geom2)}")
                print(f"    contacts[{data.ncon}]: {names}", flush=True)

set_sugar(False)
set_p9_lr(0.0, 0.0)
brain.close()
