import numpy as np
import params as x0

class DroneState:
    def __init__(self) -> None:
        self.vec = np.array([[x0.north0],  # (0) position
                               [x0.east0],   # (1)
                               [x0.down0],   # (2)
                               [x0.u0],    # (3) velocity
                               [x0.v0],    # (4)
                               [x0.w0],    # (5)
                               [x0.qw],    # (6) quat
                               [x0.qx],    # (7)
                               [x0.qy],    # (8)
                               [x0.qz],    # (9)
                               [x0.p0],    # (10) angular velocity
                               [x0.q0],    # (11)
                               [x0.r0],    # (12)
                               ])
    
    def __add__(self, delta):
        return self.vec + delta
    
    @property
    def p(self):
        return self.vec[:3]

    @property
    def v(self):
        return self.vec[3:6]

    @property
    def quat(self):
        return self.vec[6:10] #w,x,y,z

    @property
    def omega(self):
        return self.vec[10:]

        
    @property
    def R(self):
        vNorm = np.linalg.norm(self.quat[1:])
        if vNorm == 0:
            return np.eye(3)

        n = self.quat[1:]/vNorm
        theta = np.acos(self.quat[0])/2

        return np.exp(theta*self.cross(n))

    def normalize_quat(self):
        quatNorm = np.linalg.norm(self.quat)
        self.vec[6:10] /= quatNorm
        
class Drone:
    def __init__(self) -> None:
        self.state = DroneState()

        self.dt = 0.1
        self.m = 1
        self.J = np.eye(3)
        self.g = np.array([[0,0,-9.81]]).T
        self.D = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,0]])
    
    def update(self, T, tau):
        k1 = self.dynamics(self.state, T, tau)
        k2 = self.dynamics(self.state + self.dt*k1/2, T, tau)
        k3 = self.dynamics(self.state + self.dt*k2/2, T, tau)
        k4 = self.dynamics(self.state + self.dt*k3, T, tau)
        
        self.state.vec += self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.state.normalize_quat()

    def dynamics(self, state, T, tau):
        pDot = state.v
        vDot = self.g + (1/self.m)*state.R@self.aero_forces(state, T)
        quatDot = 1/2 * self.quat_action(state.quat, state.omega)
        omegaDot = np.linalg.inv(self.J) @ (-self.cross(state.omega) @ self.J@state.omega + tau)

        return np.concatenate([pDot, vDot, quatDot, omegaDot], 0)
        
    def aero_forces(self, state, T):
        # g - TRe3 - RDR^Tv
        e3 = np.array([[0,0,1]]).T
        return -T*state.R@e3 - state.R@self.D@state.R.T@state.v

    @staticmethod
    def quat_action(quat, v):
        """Rotate a 3-vector with a quaternion
        This is how the Eigen library does it
        https://gitlab.com/libeigen/eigen/-/issues/1779"""
        t = 2 * np.cross(quat[1:].T,v.T)
        return v + quat[0]*t.T + np.cross(quat[1:].T,t).T

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
        


if __name__ == "__main__":
    uav = Drone()

    thrust = 1
    torque = np.ones((3,1))
    uav.update(thrust, torque)