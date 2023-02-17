import sys
sys.path.append(".")
import numpy as np
import time

from hw2_dynamics_sim.drone import Drone
# from hw1_dummy_control.visualize import AirSimViz
from hw3_controller.trace_controller import SO3_Controller


radius = 100
def circle_traj(t):
    f = np.pi/50
    x = radius*np.sin(f * t)
    y = radius*np.cos(f * t)
    z = -100

    vx = f*radius*np.cos(f*t)
    vy = -f*radius*np.sin(f*t)
    vz = 0

    ax = -f**2*radius*np.sin(f*t)
    ay = -f**2*radius*np.cos(f*t)
    az = 0

    return np.array([[x,y,z]]).T, np.array([[vx,vy,vz]]).T, np.array([[ax,ay,az]]).T

dt = 0.01
uav = Drone(dt)
controller = SO3_Controller(dt, uav)
# viz = AirSimViz()

t = 0

while True:
    # Generate Trajectory
    pd, vd, ad = circle_traj(t)
    bd = np.array([[1,0,0]]).T

    # Trajectory Following Commands
    T, tau = controller.compute_control(uav.state, pd, vd, ad, bd)

    # Pass commands to simulation
    uav.update(T, tau)
    # viz.set_pose(uav.state.p, uav.state.quat)
    time.sleep(dt)

    t+=dt

