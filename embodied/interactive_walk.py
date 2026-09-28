"""Interactive viewer + CPG walking: the fly walks in real time; poke it with Ctrl+drag."""
import mujoco
import numpy as np

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

fly = make_locomotion_fly(name="walking_fly", add_adhesion=True, colorize=True)
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

print("Fly is walking. Drag=orbit, scroll=zoom, select body then Ctrl+drag to poke. Esc quits.")

steps_per_frame = 10
with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    while viewer.is_running():
        for _ in range(steps_per_frame):
            action = controller.step()
            apply_locomotion_action(sim, fly.name, action)
            sim.step()
        viewer.sync()
