import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        # cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        kp = np.diag(np.array([7.9, 7.9, 19]))
        kd = np.diag(np.array([5.8, 5.8, 8]))

        KR = np.diag(np.array([2500, 2550, 120]))
        KW = np.diag(np.array([115, 100, 75]))
        
        Rot = Rotation.from_quat(state['q']).as_matrix() 

        #Present State
        x = (state['x']).reshape((3,1))
        x_dot = (state['v']).reshape((3,1))
        q = state['q']
        w = state['w'].reshape((3,1))

        #Desired State
        x_des = (flat_output['x']).reshape((3,1))
        x_dot_des = (flat_output['x_dot']).reshape((3,1))
        x_ddot_des = (flat_output['x_ddot']).reshape((3,1))
        yaw_des = flat_output['yaw']

        # print('x: ', x.shape)
        # print('x_dot: ', x_dot.shape)


        #Step 1:
        x_ddot_des_acc =  x_ddot_des - (kd @ (x_dot - x_dot_des)) - (kp @ (x - x_des))
        F_des = self.mass * x_ddot_des_acc + (np.array([0, 0, self.mass * self.g]).reshape((3, 1)))

        # print('r_ddot: ', x_ddot_des_controller)
        # print('f_des: ', F_des)
        # exit()
        
        #Step 2:
        b3 = Rot @ np.array([0, 0, 1]).T
        u1 = b3.T @ F_des 

        #Step 3:
        b3_des = F_des / np.linalg.norm(F_des)
        a_si = np.array([np.cos(yaw_des), np.sin(yaw_des), 0]).reshape((3, 1))
        b2_des = np.cross(b3_des, a_si, axis=0) / np.linalg.norm(np.cross(b3_des, a_si, axis=0))
        b1_des = np.cross(b2_des, b3_des, axis=0)
        R_des = np.hstack((b1_des, b2_des, b3_des))

        # print('b3_des: ', b3_des)
        # print('b2_des: ', b2_des)
        # print('b1_des: ', b1_des)

        # R_des = Rotation.from_quat(np.array([0.3420201, 0, 0, 0.9396926])).as_matrix() #for step response in orientation
        
        #Step 4:
        er =  (R_des.T @ Rot -  Rot.T @ R_des)
        eR = 0.5 * np.array([er[2,1], er[0,2], er[1,0]]).reshape((3, 1))
        # print('eR: ', eR)
        # print('er.shape: ', er.shape)'

        #Step 5:
        # ew = 
        u2 = self.inertia @ (- KR @ eR - KW @ w)
        u = np.vstack((u1, u2))
        
        arml = self.arm_length
        gam = self.k_drag / self.k_thrust

        A = np.array([1, 1, 1, 1, 0, arml, 0, -arml, -arml, 0, arml, 0, gam, - gam, gam, -gam]).reshape((4,4))
        F = np.linalg.inv(A) @ u
        
        cmd_motor_speeds = np.sign(F) * np.sqrt(np.absolute(F) / self.k_thrust)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        # print('cmd_motor_speeds: ', cmd_motor_speeds)  
        # print('cmd_thrust: ', cmd_thrust)

        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = q

        # print(u)
        # print(F)
        # exit()

        # print('cmd_motor_speeds: ', cmd_motor_speeds)  
        # print('cmd_thrust: ', cmd_thrust)
        # print('cmd_moment: ', cmd_moment)
        # print('cmd_q: ', cmd_q)
        # print('r_ddot: ', x_ddot_des_controller)
        # print('f_des: ', F_des)
        # print('b3_des: ', b3_des)
        # print('b2_des: ', b2_des)
        # print('b1_des: ', b1_des)
        # print('R_des: ', R_des)
        # print('eR: ', eR)
        # print('u2: ', u2)
        # print('u1: ', u1)
        # print('F: ', F)
        # print('A: ', A)

        # exit()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input