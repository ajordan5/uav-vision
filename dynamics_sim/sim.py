import sys
sys.path.append(".")
import numpy as np
import time

from drone import Drone
from dummy_control.visualize import AirSimViz

dt = 0.05
uav = Drone(dt)
viz = AirSimViz()

m = 10
thrust = m*9.81*0
T = thrust/m
mx = 0
my = 0
mz = -0.1
tau = np.array([[mx, my, mz]]).T

while True:
    uav.update(T, tau)
    viz.set_pose(uav.state.p, uav.state.quat)
    time.sleep(dt)

