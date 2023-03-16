import cv2
import numpy as np
from tools.dirty_derivative import DirtyDerivative

class TimeToCollision:
    def __init__(self) -> None:
        self.first_image = True
        self.p0 = None
        self.prev_grey = None
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Create random colors
        self.color = np.random.randint(0, 255, (100, 3))

    def compute_time(self, image):
        # Create a mask image for drawing purposes
        mask = np.zeros_like(image)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if self.first_image:
            self.p0 = cv2.goodFeaturesToTrack(grey, mask=None, **self.feature_params)
            self.first_image = False
            self.prev_grey = grey
            return
        
        if len(self.p0) < 30:
            self.p0 = cv2.goodFeaturesToTrack(grey, mask=None, **self.feature_params)


        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_grey, grey, self.p0, None, **self.lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        print(len(self.p0))

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.astype(int).ravel()
            c, d = old.astype(int).ravel()
            
            mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
            image = cv2.circle(image, (a, b), 5, self.color[i].tolist(), -1)
    
        # Display the demo
        img = cv2.add(image, mask)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF

        # Update the previous frame and previous points
        self.prev_grey = grey.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        
            

        
        

