import airsim
import numpy as np
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# while True:

for t in np.arange(0, 100, 0.01):
    x = 20*np.sin(t)
    y = 20*np.cos(t)
    z = -30
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x,y,z), airsim.to_quaternion(0,0,0)), True)
    time.sleep(0.05)