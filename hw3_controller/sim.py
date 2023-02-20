import sys
sys.path.append(".")
import numpy as np
import time

from hw2_dynamics_sim.drone import Drone
from hw1_dummy_control.visualize import AirSimViz
from hw3_controller.trace_controller import SO3_Controller, MsgTrajectory
from hw3_controller.data_viewer import DataViewer


radius = 3
def circle_traj(t):
    f = np.pi/3
    x = radius*np.sin(f * t)
    y = radius*np.cos(f * t) 
    z = -100

    vx = f*radius*np.cos(f*t)
    vy = -f*radius*np.sin(f*t)
    vz = 0

    ax = -f**2*radius*np.sin(f*t)
    ay = -f**2*radius*np.cos(f*t)
    az = 0

    msg = MsgTrajectory()
    msg.set(np.array([[x,y,z]]).T, np.array([[vx,vy,vz]]).T, np.array([[ax,ay,az]]).T, 0)

    return msg

dt = 0.005
visualize = 1

dataView = DataViewer()
uav = Drone(dt)
controller = SO3_Controller(dt, uav)

if visualize:
    viz = AirSimViz()

t = 0

while True:
    # Generate Trajectory
    traj = circle_traj(t)
    psid = 0

    # Trajectory Following Commands
    T, tau = controller.compute_control(uav.state, traj)

    # Pass commands to simulation
    uav.update(T, tau)
    # print(np.concatenate((uav.state.p, pd),1))

    # Visualize in Airsim and plots
    dataView.update(uav.state, uav.state, traj, (T,tau), dt)
    # print(uav.state.p)
    if visualize:
        viz.set_pose(uav.state.p, uav.state.quat)

    # time.sleep(dt)
    t+=dt

