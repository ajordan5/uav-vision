import cv2
import sys
sys.path.append(".")
from path_gen.pixel_to_ned import px_to_ned
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class TravelingSalesman:
    def __init__(self, num_pts=10, altitude=-5) -> None:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.on_mouse)
        self.waypoints = []
        self.alt = altitude
        
        self.img = np.ones((1000,1000,3))*255
        
        while len(self.waypoints) < num_pts:
            cv2.imshow('image',self.img)
            k = cv2.waitKey(10)
            if k == ord('x'):
                break
        cv2.destroyAllWindows()


        self.traj = BSplineTrajectory()
        self.traj.find_path_through_waypoints(np.array(self.pixel_waypoints))        

            

    def on_mouse(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.img = cv2.circle(self.img, (x,y), 5, (255,0,0), 2)
            self.waypoints.append([x,y])


class BSplineTrajectory:
    def __init__(self, tf=5, dt=0.2, order=3) -> None:
        self.knots = np.arange(0, tf+dt, dt, )
        self.num_ctrl_pts = len(self.knots)
        self.order=order
        self.tf = tf

        for i in range(order):
            self.knots = np.append(self.knots,tf)
            self.knots = np.insert(self.knots, 0, 0)

    def find_path_through_waypoints(self, waypoints):

        # Get distances between waypoints to calculate times
        distances = []
        total = 0
        for i, w in enumerate(waypoints):
            if i==0:
                continue

            diff = np.linalg.norm(w - waypoints[i-1])
            total +=diff
            distances.append(total)

        wp_times = np.array(distances)/total * self.tf

        self.start_pt = waypoints[0,:].reshape(1,-1)
        self.end_pt = waypoints[-1,:].reshape(1,-1)
        middle = waypoints[1:-1,:]

        interp = np.linspace(0,1,self.num_ctrl_pts)
        ctrl_pts0 = (self.start_pt + (self.end_pt-self.start_pt)*interp[:,None])[:,:2]
        # self.plot_traj(ctrl_pts0)
        
        # Set the maximum number of iterations to 100
        max_iterations = 1000

        # Set the options parameter
        options = {'maxiter': max_iterations}
        cons_ineq = [{'type':'ineq', 'fun':self.wp_constraint, 'args': (middle, wp_times[:-1])}]
        opt_ctrl_pts = minimize(self.objective, ctrl_pts0, method='SLSQP', options=options, constraints=cons_ineq)
        print(opt_ctrl_pts)
        self.plot_traj(opt_ctrl_pts.x, wp=middle)

    def objective(self, ctrl_pts, minimize_derivative=2):
        ctrl_pts = ctrl_pts.reshape((-1,2))
        altitude = np.ones((ctrl_pts.shape[0],1))*self.start_pt[0,2]
        ctrl_pts = np.concatenate((ctrl_pts,altitude),1)

        all_pts = np.concatenate((self.start_pt, ctrl_pts, self.end_pt), 0)
        path = BSpline(t=self.knots, c=all_pts, k=self.order)
        derivative_objective = path.derivative(minimize_derivative)

        # Integrate snap across knots
        total_accel = 0
        for t in np.arange(self.knots[0], self.knots[-1], 0.2):
            total_accel += np.linalg.norm(derivative_objective(t))

        return total_accel

    def wp_constraint(self, ctrl_pts, wp, wp_ts, tol=1e-0):
        ctrl_pts = ctrl_pts.reshape((-1,2))
        altitude = np.ones((ctrl_pts.shape[0],1))*self.start_pt[0,2]
        ctrl_pts = np.concatenate((ctrl_pts,altitude),1)

        all_pts = np.concatenate((self.start_pt, ctrl_pts, self.end_pt), 0)
        path = BSpline(t=self.knots, c=all_pts, k=self.order)

        cons = []
        for idx, pt in enumerate(wp):
            t = wp_ts[idx]
            dist = np.linalg.norm(path(t) - pt)
            cons.append(tol - dist)
        # print(cons)
        return np.array(cons)
    
    def plot_traj(self, middle_pts, wp=None):
        middle_pts = middle_pts.reshape((-1,2))
        altitude = np.ones((middle_pts.shape[0],1))*self.start_pt[0,2]
        middle_pts = np.concatenate((middle_pts,altitude),1)

        ctrl_pts = np.concatenate((self.start_pt, middle_pts, self.end_pt), 0)
        spl = BSpline(t=self.knots, c=ctrl_pts, k=self.order)

        t0 = spl.t[0]  # first knot is t0
        tf = spl.t[-1]  # last knot is tf
        # number of points in time vector so spacing is 0.01
        N = int(np.ceil((tf - t0)/0.01))
        t = np.linspace(t0, tf, N)  # time vector
        position = spl(t)
        # 3D trajectory plot
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(spl.c[:, 0], spl.c[:, 1], spl.c[:, 2],
                '-o', label='control points')
        ax.plot(position[:, 0], position[:, 1], position[:, 2],
                'b', label='spline')
        
        # ax.scatter(self.start_pt, label="Start")
        ax.scatter(self.end_pt[0,0], self.end_pt[0,1], self.end_pt[0,2], color='g', label="End")
        ax.scatter(self.start_pt[0,0], self.start_pt[0,1], self.start_pt[0,2], color='m', label="Start")
        if wp is not None:
            ax.scatter(wp[:,0], wp[:,1], wp[:,2], color='r', label = "waypoints")
        #ax.set_xlim3d([-10, 10])
        ax.legend()
        ax.set_xlabel('x', fontsize=16, rotation=0)
        # ax.axes.set_xlim3d(0,10)
        # ax.axes.set_ylim3d(0,10)
        # ax.axes.set_zlim3d(0,10)
        ax.set_ylabel('y', fontsize=16, rotation=0)
        ax.set_zlabel('z', fontsize=16, rotation=0)
        plt.show()


        


if __name__ == "__main__":
    path = TravelingSalesman()

    


