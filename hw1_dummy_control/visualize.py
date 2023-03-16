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


    def get_image(self):
        im = self.client.simGetImages([airsim.ImageRequest("bottom_center_custom", airsim.ImageType.Scene, False, False)])[0]
        # get numpy array
        img1d = np.fromstring(im.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(im.height, im.width, 3)

        # original image is fliped vertically
        # img_rgb = np.flipud(img_rgb)

        return img_rgb
