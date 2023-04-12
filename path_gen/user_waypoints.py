import cv2
import sys
sys.path.append(".")
from path_gen.pixel_to_ned import px_to_ned
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class UserDefinedPath:
    def __init__(self, num_pts=10, altitude=-5) -> None:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.on_mouse)
        self.waypoints = []
        self.pixel_waypoints = []
        self.alt = altitude

        img = cv2.imread("./path_gen/altitude_map.png")
        
        while len(self.waypoints) < num_pts:
            cv2.imshow('image',img)
            k = cv2.waitKey(10)
            if k == ord('x'):
                break
        cv2.destroyAllWindows()


        self.traj = BSplineTrajectory()
        self.traj.find_path_through_waypoints(np.array(self.pixel_waypoints))
        self.best_path = self.traj.spl
        # self.traj = BSplineTrajectory(num_segments=10)        
        # self.traj.find_path_through_waypoints(np.array(self.pixel_waypoints))        

            

    def on_mouse(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            north, east = px_to_ned(x,y)
            self.waypoints.append([north,east,self.alt])
            self.pixel_waypoints.append([x,y,0])


class BSplineTrajectory:
    def __init__(self, tf=1, num_segments=20, order=3) -> None:
        self.knots = self.uniform_knots(tf, order, num_segments)
        self.N = num_segments+order
        self.order=order
        self.tf = tf


    def uniform_knots(self, tf, order, num_segments):
        t0=0
        knots = np.concatenate((
            t0 * np.ones(order),
            np.concatenate((
                np.arange(t0, tf, (tf-t0)/(num_segments)),
                tf * np.ones(order+1)),
                axis=0)
            ), axis=0)
        return knots

    def get_waypoint_times(self):
        distances = []
        total = 0
        for i, w in enumerate(self.waypoints):
            if i==0:
                continue

            diff = np.linalg.norm(w - self.waypoints[i-1])
            total +=diff
            distances.append(total)

        return np.array(distances)/total * self.tf
    

    def setup_cps(self):
        # Start and end at the first and last wp
        self.start_pt = self.waypoints[0,:].reshape(1,-1)
        self.end_pt = self.waypoints[-1,:].reshape(1,-1)

        # Clamp accel and vel to start at zero
        self.start_pts = np.tile(self.waypoints[0,:].reshape(1,-1), (3,1))
        self.end_pts = np.tile(self.waypoints[-1,:].reshape(1,-1), (3,1))
        self.middle = self.waypoints[1:-1,:]

        interp = np.linspace(0,1,self.N-6)
        self.ctrl_pts0 = (self.start_pt + (self.end_pt-self.start_pt)*interp[:,None])[:,:2]
        

    def find_path_through_waypoints(self, waypoints):
        self.waypoints = waypoints

        # Get distances between waypoints to calculate times
        
        wp_times = self.get_waypoint_times()
        self.setup_cps()

        
        # self.plot_traj(ctrl_pts0)
        
        # Set the maximum number of iterations to 1000
        max_iterations = 1000

        # Set the options parameter
        options = {'maxiter': max_iterations}
        cons_ineq = [{'type':'ineq', 'fun':self.wp_constraint, 'args': (self.middle, wp_times[:-1])}]
        opt_ctrl_pts = minimize(self.objective, self.ctrl_pts0, method='SLSQP', options=options, constraints=cons_ineq)
        print(opt_ctrl_pts)

        # Put together optimal path
        middle_pts = opt_ctrl_pts.x.reshape((-1,2))
        altitude = np.ones((middle_pts.shape[0],1))*self.start_pt[0,2]
        middle_pts = np.concatenate((middle_pts,altitude),1)

        ctrl_pts = np.concatenate((self.start_pts, middle_pts, self.end_pts), 0)
        self.spl = BSpline(t=self.knots, c=ctrl_pts, k=self.order)
        self.plot_traj()

    def objective(self, ctrl_pts, minimize_derivative=2):
        ctrl_pts = ctrl_pts.reshape((-1,2))
        altitude = np.ones((ctrl_pts.shape[0],1))*self.start_pt[0,2]
        ctrl_pts = np.concatenate((ctrl_pts,altitude),1)

        all_pts = np.concatenate((self.start_pts, ctrl_pts, self.end_pts), 0)
        path = BSpline(t=self.knots, c=all_pts, k=self.order)
        derivative_objective = path.derivative(minimize_derivative)

        # Integrate snap across knots
        total_accel = 0
        t = np.linspace(self.knots[0], self.knots[-1], self.N)
        accel = derivative_objective(t)


        return np.linalg.norm(accel)

    def wp_constraint(self, ctrl_pts, wp, wp_ts, tol=1e-0):
        ctrl_pts = ctrl_pts.reshape((-1,2))
        altitude = np.ones((ctrl_pts.shape[0],1))*self.start_pt[0,2]
        ctrl_pts = np.concatenate((ctrl_pts,altitude),1)

        all_pts = np.concatenate((self.start_pts, ctrl_pts, self.end_pts), 0)
        path = BSpline(t=self.knots, c=all_pts, k=self.order)

        cons = []
        for idx, pt in enumerate(wp):
            t = wp_ts[idx]
            dist = np.linalg.norm(path(t) - pt)
            cons.append(tol - dist)
        # print(cons)
        return np.array(cons)
    
    def plot_traj(self):
        

        t0 = self.spl.t[0]  # first knot is t0
        tf = self.spl.t[-1]  # last knot is tf
        # number of points in time vector so spacing is 0.01
        N = int(np.ceil((tf - t0)/0.01))
        t = np.linspace(t0, tf, N)  # time vector
        position = self.spl(t)
        # 3D trajectory plot
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.spl.c[:, 0], self.spl.c[:, 1], self.spl.c[:, 2],
                '-o', label='control points')
        ax.plot(position[:, 0], position[:, 1], position[:, 2],
                'b', label='spline')
        
        # ax.scatter(self.start_pt, label="Start")
        ax.scatter(self.end_pt[0,0], self.end_pt[0,1], self.end_pt[0,2], color='g', label="End")
        ax.scatter(self.start_pt[0,0], self.start_pt[0,1], self.start_pt[0,2], color='m', label="Start")
        if self.waypoints is not None:
            ax.scatter(self.waypoints[:,0], self.waypoints[:,1], self.waypoints[:,2], color='r', label = "waypoints")
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
    path = UserDefinedPath()

    


