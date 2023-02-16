import numpy as np

north0 = 0.  # initial north position
east0 = 0.  # initial east position
down0 = -100.0  # initial down position
u0 = 0.  # initial velocity along body x-axis
v0 = 0.  # initial velocity along body y-axis
w0 = 0.  # initial velocity along body z-axis
p0 = 0  # initial roll rate
q0 = 0  # initial pitch rate
r0 = 0  # initial yaw rate

#   Quaternion State
qw = 1
qx = 0
qy = 0
qz = 0

# Physical params
m = 1
J = np.eye(3)*0.1
Cd = 0.1

# Controller Gains
kx = 0.1
kv = 0.5
kR = 0.3
kOmega = 0.1