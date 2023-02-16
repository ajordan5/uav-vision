import sys
sys.path.append(".")

import numpy as np
import math
from dynamics_sim.drone import Drone

def test_one_step():
    dt = 0.05
    uav = Drone(dt)

    thrust = 10
    torque = np.array([[0.1, 0.1, 0.1]]).T

    uav.update(thrust, torque)

    expectedState = np.array([-2.60619203e-06,  2.60456367e-06, -1.00000237e+02, -2.06019858e-04,
                                2.05825992e-04, -9.49971312e-03,  9.99999414e-01,  6.24999878e-04,
                                6.24999878e-04,  6.24999878e-04,  5.00000000e-02,  5.00000000e-02,
                                5.00000000e-02])
    
    for idx, val in enumerate(expectedState):
        np.testing.assert_almost_equal(uav.state.vec[idx][0], val, 6)

