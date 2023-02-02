import numpy as np

class Drone:
    def __init__(self) -> None:
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.omega = np.zeros(3)
        self.quat = np.array([1,0,0,0]) # w,x,y,z

        self.m = 1
        self.J = np.eye(3)
        self.g = np.array([[0,0,-9.81]]).T

    @property
    def R(self):
        vNorm = np.linalg.norm(self.quat[1:])
        if vNorm == 0:
            return np.eye(3)

        n = self.quat[1:]/vNorm
        theta = np.acos(self.quat[0])/2

        return np.exp(theta*self.cross(n))

    def dynamics(self, T, tau):
        pDot = self.v
        vDot = self.g + (1/self.m)*self.R@self.aero_forces()
        quatDot = 1/2 * self.quat_action(self.quat, self.omega)
        omegaDot = 1/self.J * (-self.omega @ self.cross(self.J@self.omega) + tau)

        return pDot, vDot, quatDot, omegaDot
        
    def aero_forces(self, T):
        # g - TRe3 - RDR^Tv
        return self.g - T@self.R@self.e3 - self.R@self.D@self.R.T@self.v

    @staticmethod
    def quat_action(quat, v):
        """Rotate a 3-vector with a quaternion
        This is how the Eigen library does it
        https://gitlab.com/libeigen/eigen/-/issues/1779"""
        t = 2 * np.cross(quat[1:],v)
        return v + quat[0]*t + np.cross(quat[1:],t)

    @staticmethod
    def cross(x):
        """Moves a 3 vector into so(3)

        Args:
            x (3 ndarray) : Parametrization of Lie Algebra

        Returns:
            x (3,3 ndarray) : Element of so(3)"""
        return np.array([[   0, -x[2],  x[1]],
                        [ x[2],     0, -x[0]],
                        [-x[1],  x[0],     0]])
        


