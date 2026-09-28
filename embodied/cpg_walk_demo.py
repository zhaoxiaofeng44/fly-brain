"""CPG walking demo adapted from flygym tutorial 4a (v2.1.0 API)."""
from pathlib import Path

import numpy as np
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)
from flygym.utils.math import Rotation3D

output_dir = Path(__file__).parent / "demo_output"
output_dir.mkdir(parents=True, exist_ok=True)

fly = make_locomotion_fly(name="cpg_demo", add_adhesion=True, colorize=True)
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
controller = CPGController(
    cpg_network=cpg_network,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)
print("Actuated DoFs:", len(dof_order))
print("Timestep:", sim.timestep)

sim.reset()
initial_action = LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool),
)
apply_locomotion_action(sim, fly.name, initial_action)
sim.warmup()

run_time = 2.0
nsteps_sim = int(run_time / sim.timestep)
time_grid = np.arange(nsteps_sim) * sim.timestep
thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
thorax_positions = np.full((nsteps_sim, 3), np.nan, dtype=np.float32)

for step_idx in trange(nsteps_sim, desc="Running CPG"):
    action = controller.step()
    apply_locomotion_action(sim, fly.name, action)
    sim.step_with_profile()
    thorax_positions[step_idx] = sim.get_body_positions(fly.name)[thorax_idx]
    sim.render_as_needed_with_profile()

print("Rendered frames:", len(renderer.frames[body_cam.name]))
disp = thorax_positions[-1, 0] - thorax_positions[0, 0]
print(f"Thorax forward displacement: {disp:.2f} mm in {run_time}s")

sim.renderer.save_video(output_dir / "cpg_walk.mp4")
np.save(output_dir / "thorax_positions.npy", thorax_positions)
print("Saved:", output_dir / "cpg_walk.mp4")
