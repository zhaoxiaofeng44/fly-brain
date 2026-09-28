"""Launch the MuJoCo interactive viewer with the NeuroMechFly locomotion model."""
from flygym import Simulation, launch_interactive_viewer
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

fly = make_locomotion_fly(name="interactive_fly", add_adhesion=True, colorize=True)
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))

sim = Simulation(world)
print("Launching interactive viewer. Drag to orbit, scroll to zoom, press Esc to quit.")
launch_interactive_viewer(sim.mj_model, sim.mj_data)
