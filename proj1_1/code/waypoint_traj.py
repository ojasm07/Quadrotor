import numpy as np

class WaypointTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.points = points
        self.velocity = 2.6
        
    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0


        if (len(self.points) == 1):
            x = self.points[0]
            x_dot = np.zeros((3,))
            x_ddot = np.zeros((3,))
            x_dddot = np.zeros((3,))
            x_ddddot = np.zeros((3,))
            yaw = 0
            yaw_dot = 0
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                            'yaw':yaw, 'yaw_dot':yaw_dot }
            return flat_output
        else:
            dist_seg = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)[:, np.newaxis]
            unit_vectors = (self.points[1:] - self.points[:-1]) / dist_seg   
            time_dur = dist_seg / self.velocity     

            t_start = np.zeros((self.points.shape[0], 1))

            for i in range(1, self.points.shape[0]):
                t_start[i] = t_start[i-1] + time_dur[i-1]

            for i in range(len(time_dur)):
                
                if t >= t_start[i] and t < t_start[i+1]:
                    x_dot = self.velocity * unit_vectors[i]
                    x = self.points[i] + x_dot * (t - t_start[i])
                    break

                elif t >= t_start[-1]:
                    x = self.points[-1]
                    x_dot = np.zeros((3,))
                    break 

            # print(flat_output)
            flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                            'yaw':yaw, 'yaw_dot':yaw_dot }
            # print(flat_output)  
            return flat_output