import numpy as np
import dynamics_sim.params as x0
from dynamics_sim.utils import *

class DroneState:
    def __init__(self, init=None) -> None:
        if init is None:
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
        else:
            self.vec = init

    def __add__(self, delta):
        
        return DroneState(self.vec+delta)
    
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
        e0,e1,e2,e3 = self.quat.flatten()

        R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                    [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                    [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
        R = R/np.linalg.det(R)

        return R

    def normalize_quat(self):
        quatNorm = np.linalg.norm(self.quat)
        self.vec[6:10] /= quatNorm
        
class Drone:
    def __init__(self, dt=0.1) -> None:
        self.state = DroneState()

        self.dt = dt
        self.m = x0.m
        self.J = x0.J
        self.g = np.array([[0,0,9.81]]).T
        self.D = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,0]]) * x0.Cd * 9.81
    
    def update(self, T, tau):
        k1 = self.dynamics(self.state, T, tau)
        k2 = self.dynamics(self.state + self.dt*k1/2, T, tau)
        k3 = self.dynamics(self.state + self.dt*k2/2, T, tau)
        k4 = self.dynamics(self.state + self.dt*k3, T, tau)
        
        self.state.vec += self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.state.normalize_quat()

    def dynamics(self, state, T, tau):
        pDot = state.v
        vDot = self.g + self.aero_forces(state, T)

        omegaBar = np.insert(state.omega, 0, 0)
        quatDot = 1/2 * quat_action(state.quat, omegaBar)
        omegaDot = np.linalg.inv(self.J) @ (-cross(state.omega) @ (self.J@state.omega) + tau)

        return np.concatenate([pDot, vDot, quatDot, omegaDot], 0)
        
    def aero_forces(self, state, T):
        # g - TRe3 - RDR^Tv
        e3 = np.array([[0,0,1]]).T
        return -T*state.R@e3 - state.R@self.D@state.R.T@state.v

    def mav_quat_dot(self, quat, omega):
        e0,e1,e2,e3 = quat
        p,q,r = omega
        e0_dot = 0.5 * (-p*e1 - q*e2 - r*e3)
        e1_dot = 0.5 * (p*e0 + r*e2 - q*e3)
        e2_dot = 0.5 * (q*e0 - r*e1 + p*e3)
        e3_dot = 0.5 * (r*e0 + q*e1 -p*e2)
        return np.array([e0_dot, e1_dot, e2_dot, e3_dot])

    
        


if __name__ == "__main__":
    uav = Drone()
    print("x0", uav.state.vec)

    thrust = 1
    torque = np.ones((3,1))
    uav.update(thrust, torque)
    print("x1", uav.state.vec)