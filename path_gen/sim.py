import sys
sys.path.append(".")
import numpy as np
import time
import cv2

from hw2_dynamics_sim.drone import Drone
from hw1_dummy_control.visualize import AirSimViz
from hw5_optical_flow.velocity_controller import SO3_Controller, MsgTrajectory
from hw5_optical_flow.optical_flow import TimeToCollision


radius = 3
def hold_traj(t):
    f = np.pi/3
    x = 0
    y = 0
    z = 0

    vx = 0
    vy = 0
    vz = 0

    ax = 0
    ay = 0
    az = 0

    msg = MsgTrajectory()
    msg.set(np.array([[x,y,z]]).T, np.array([[vx,vy,vz]]).T, np.array([[ax,ay,az]]).T, 0)

    return msg

dt = 0.05
visualize = 1

# dataView = DataViewer()
uav = Drone(dt)
controller = SO3_Controller(dt, uav)

if visualize:
    viz = AirSimViz()

flow = TimeToCollision()

t = 0

while True:
    # Generate Trajectory
    traj = hold_traj(t)
    psid = 0

    # Trajectory Following Commands
    T, tau = controller.compute_control(uav.state, traj)

    # Pass commands to simulation
    uav.update(T, tau)

    # print(uav.state.p)
    if visualize:
        viz.set_pose(uav.state.p, uav.state.quat)

    t5 = time.time()

    # print(t2-t1, t3-t2, t4-t3, t5-t4, t5-t1, t)
    # time.sleep(dt)
    t+=dt
    # print(t)

