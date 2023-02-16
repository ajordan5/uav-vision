import sys
sys.path.append(".")
import numpy as np
import time

from hw2_dynamics_sim.drone import Drone
from hw1_dummy_control.visualize import AirSimViz
from hw3_controller.trace_controller import SO3_Controller

dt = 0.01
uav = Drone(dt)
controller = SO3_Controller(dt, uav)
viz = AirSimViz()

while True:
    # generate trajectory in one message?
    T, tau = controller.compute_control()
    uav.update(T, tau)
    viz.set_pose(uav.state.p, uav.state.quat)
    time.sleep(dt)

