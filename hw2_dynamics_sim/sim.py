import sys
sys.path.append(".")
import numpy as np
import time

from drone import Drone
from hw1_dummy_control.visualize import AirSimViz

dt = 0.01
uav = Drone(dt)
viz = AirSimViz()

m = 10
thrust = m*9.81*1.1
T = thrust/m
mx = 0.01
my = 0.0
mz = -0.0
tau = np.array([[mx, my, mz]]).T

while True:
    uav.update(T, tau)
    viz.set_pose(uav.state.p, uav.state.quat)
    time.sleep(dt)

