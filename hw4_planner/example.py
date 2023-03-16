"""
landing trajectory with minimum acceleration / jerk
        2/17/22 - RWB
"""
import numpy as np
from math import ceil
from scipy.interpolate import BSpline
from scipy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def plotSpline(spl, wp=None):
    t0 = spl.t[0]  # first knot is t0
    tf = spl.t[-1]  # last knot is tf
    # number of points in time vector so spacing is 0.01
    N = ceil((tf - t0)/0.01)
    t = np.linspace(t0, tf, N)  # time vector
    position = spl(t)
    # 3D trajectory plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # plot control points (convert YX(-Z) -> NED)
    ax.plot(spl.c[:, 1], spl.c[:, 0], -spl.c[:, 2],
            '-o', label='control points')
    # plot spline (convert YX(-Z) -> NED)
    ax.plot(position[:, 1], position[:, 0], -position[:, 2],
            'b', label='spline')
    

    if wp is not None:
        ax.scatter(wp[:,1], wp[:,0], -wp[:,2], color='r', label = "waypoints")
    #ax.set_xlim3d([-10, 10])
    ax.legend()
    ax.set_xlabel('x', fontsize=16, rotation=0)
    ax.set_ylabel('y', fontsize=16, rotation=0)
    ax.set_zlabel('z', fontsize=16, rotation=0)
    plt.show()

def accel_fun(ctrl_pts, start, end, knots, order):
    ctrl_pts = ctrl_pts.reshape((-1,3))
    all_pts = np.concatenate((start, ctrl_pts, end), 0)
    path = BSpline(t=knots, c=all_pts, k=order)
    accel = path.derivative(2)

    # Integrate snap across knots
    total_accel = 0
    for t in np.arange(knots[0], knots[-1], 0.1):
        total_accel += np.linalg.norm(accel(t))

    return total_accel

def waypoint_fun(ctrl_pts, start, end, knots, order, wp, wp_ts):
    ctrl_pts = ctrl_pts.reshape((-1,3))
    all_pts = np.concatenate((start, ctrl_pts, end), 0)
    path = BSpline(t=knots, c=all_pts, k=order)

    total_dist = 0
    for idx, pt in enumerate(wp):
        t = wp_ts[idx]
        total_dist += np.linalg.norm(path(t) - pt)


    accel = path.derivative(2)

    # Integrate snap across knots
    total_accel = 0
    for t in np.arange(knots[0], knots[-1], 0.1):
        total_accel += np.linalg.norm(accel(t))

    return 100*total_dist + total_accel

if __name__ == "__main__":
    # initial and final time
    t0 = 0
    tf = 5
    order = 3
    knots = np.array([t0, t0, t0, t0,
                      (tf-t0)/3, 2*(tf-t0)/3,
                      tf, tf, tf, tf])
    # num control = num knots - order - 1
    start_pt = np.array([[0, 0, 0]])
    end_pt = np.array([[2, 1, 0]])
    
    ctrl_pts = np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0]])
    

    # Min accel trajectory, just a straight line
    # res = minimize(accel_fun, ctrl_pts, args=(start_pt, end_pt, knots, order), method='SLSQP')
    # print(res)

    # cp_star = res.x.reshape((-1,3))
    # all_pts = np.concatenate((start_pt, cp_star, end_pt), 0)
    # spl = BSpline(t=knots, c=all_pts, k=order)
    # plotSpline(spl)

    # Min distance from waypoint at ti
    waypoints = np.array([[0.5,0.5,0],
                          [0.75,0,0]])
    waypoint_ts = [1,3]
    res = minimize(waypoint_fun, ctrl_pts, args=(start_pt, end_pt, knots, order, waypoints, waypoint_ts), method='SLSQP')
    print(res)

    cp_star = res.x.reshape((-1,3))
    all_pts = np.concatenate((start_pt, cp_star, end_pt), 0)
    spl = BSpline(t=knots, c=all_pts, k=order)
    plotSpline(spl, wp=waypoints)