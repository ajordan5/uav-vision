import numpy as np
import hw2_dynamics_sim.params as gains
from tools.utils import *
from tools.dirty_derivative import DirtyDerivative

class MsgTrajectory:
    def __init__(self):
        self.pos = np.zeros((3, 1))  # commanded position in m
        self.vel = np.zeros((3, 1))  # commanded velocity in m/s
        self.accel = np.zeros((3, 1))  # commanded acceleration in m/s/s
        self.heading = 0.0  # commanded heading in rad
        self.R = np.eye(3)

    def set(self, p=np.zeros((3,1)), v=np.zeros((3,1)), a=np.zeros((3,1)), psi=0.0):
        self.pos = p
        self.vel = v
        self.a = a
        self.heading = 0


class SO3_Controller:
    def __init__(self, dt, drone) -> None:
        self.drone = drone
        self.omegadot = DirtyDerivative(dt, 5*dt)
        self.Rdot = DirtyDerivative(dt, 5*dt)

    def compute_control(self, state, traj):
        ex = 0
        ev = state.v - traj.vel

        # Get desired attitude
        thrustComponent = -gains.kx*ex - gains.kv*ev - self.drone.m*self.drone.g
        kd = -thrustComponent / np.linalg.norm(thrustComponent)
        # b2 = cross(b3)@b1
        # Rd = np.concatenate((b1,b2,b3)) # Not orthonormal??
        sd = np.array([[np.cos(traj.heading)], [np.sin(traj.heading)], [0.]])
        jd = cross(kd) @ sd
        jd = jd / np.linalg.norm(jd)
        id = cross(jd) @ kd
        Rd = np.concatenate((id, jd, kd), axis=1)
        traj.R = Rd
        
        eR = 0.5 * vee(Rd.T@state.R - state.R.T@Rd)

        RdDot = self.Rdot.update(Rd)
        omegad = vee(0.5*(Rd.T@RdDot - RdDot.T@Rd)) # jake's code line 73 is different TODO
        omegadDot = self.omegadot.update(omegad)

        eOmega = state.omega - state.R.T@Rd@omegad

        T = self.saturate(np.dot(-(thrustComponent).T, state.R[:,2]))
        tau = -gains.kR*eR - gains.kOmega*eOmega + cross(state.omega) @ self.drone.J@state.omega \
            - self.drone.J @ (cross(state.omega)@state.R.T@Rd@omegad - state.R.T@Rd@omegadDot)

        return T, tau


    def saturate(self, T):
        if T > 100:
            return 100
        elif T < 0:
            return 0

        return T
        
