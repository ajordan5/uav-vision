import numpy as np


def vee(M):
    """
    Maps skew-symmetric matrix to a vector
    """
    if np.linalg.norm(M+M.T) != 0:
        print("M is not skew-symmetric")
        m = float("nan")
    else:
        m = np.array([[M[2][1]], [-M[2][0]], [M[1][0]]])
    return m

def quat_action(quat1, quat2):
        """Multiply 2 quaternions"""
        w0, x0, y0, z0 = quat1
        w1, x1, y1, z1 = quat2
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])

def cross(x):
    """Moves a 3 vector into so(3)

    Args:
        x (3 ndarray) : Parametrization of Lie Algebra

    Returns:
        x (3,3 ndarray) : Element of so(3)"""
    x = x.flatten()
    return np.array([[   0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],     0]])