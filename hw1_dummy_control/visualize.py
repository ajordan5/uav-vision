import airsim
import numpy as np
import time

class AirSimViz:
    def __init__(self) -> None:
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    
    def set_pose(self, position, quat):
        pose = self.client.simGetVehiclePose("rover")
        x,y,z = position.flatten()
        qw,qx,qy,qz = quat.flatten()

        pose.position.x_val = x
        pose.position.y_val = y
        pose.position.z_val = z

        pose.orientation.x_val = qx
        pose.orientation.y_val = qy
        pose.orientation.z_val = qz
        pose.orientation.w_val = qw

        self.client.simSetVehiclePose(pose, False)

