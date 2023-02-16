import numpy as np
import hw2_dynamics_sim.params as gains
from tools.utils import *
from tools.dirty_derivative import DirtyDerivative

class SO3_Controller:
    def __init__(self, dt, drone) -> None:
        self.drone = drone
        self.omegadot = DirtyDerivative(dt, 5*dt)
        self.Rdot = DirtyDerivative(dt, 5*dt)

    def compute_control(self, state, xd, vd, ad, b1):
        ex = state.p - xd
        ev = state.v - vd

        # Get desired attitude
        thrustComponent = -gains.kx*ex - gains.kv*ev - self.drone.m*self.drone.g + self.drone.m*ad
        b3 = thrustComponent / np.linalg.norm(thrustComponent)
        b2 = cross(b3)@b1
        Rd = np.concatenate((b1,b2,b3))
        eR = 0.5 * vee(Rd.T@state.R - state.R.T@Rd)

        RdDot = self.Rdot.update(Rd)
        omegad = vee(Rd.T@RdDot) # jake's code line 73 is different TODO
        omegadDot = self.omegadot.update(omegad)

        eOmega = state.omega - state.R.T@Rd@omegad

        T = -(thrustComponent) @ state.R[:,2]
        tau = -gains.kR*eR - gains.kOmega*eOmega + cross(state.omega) @ self.drone.J@state.omega \
            - self.drone.J @ (cross(state.omega)@state.R.T@Rd@omegad - state.R.T@Rd@omegadDot)

        return T, tau
