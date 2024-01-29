#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    # print(q)
    # exit()

    R = Rotation.as_matrix(q)
    # print('R:', R.shape)
    # print('a_m:', a_m.shape)
    # print('a_b:', a_b.shape)
    # print('g:', g.shape)
    # print('w_m:', w_m.shape)
    # print('w_b:', w_b.shape)
    # print('dt:', dt.shape)
    # print('p:', p.shape)
    # print('v:', v.shape)
    # print('q:', q.shape)

    new_p = p + v*dt + 0.5*(R@(a_m - a_b) + g)*(dt**2)
    new_v = v + (R@(a_m - a_b) + g)*dt
    new_q = q * Rotation.from_rotvec(((w_m - w_b)*dt).flatten())

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = Rotation.as_matrix(q)

    # print('error state cov: ', error_state_covariance)
    # exit()
    # print('w_m: ', w_m.shape)
    # print('a_m: ', a_m.shape)
    # print('dt: ', dt.shape)
    # print('accelerometer_noise_density: ', accelerometer_noise_density)
    # print('gyroscope_noise_density: ', gyroscope_noise_density)
    # print('accelerometer_random_walk: ', accelerometer_random_walk)
    # print('gyroscope_random_walk: ', gyroscope_random_walk)

    # exit()
    w = a_m - a_b
    skew_mat = np.array([0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0], dtype=object).reshape((3,3))
    Idt = np.eye(3)*dt

    Fx = np.eye(18)
    # First row
    Fx[0:3, 3:6] = Idt

    # Second row
    # Fx[3:6, 3:6] = np.eye(3)
    Fx[3:6, 6:9] = - (R @ skew_mat) * dt
    Fx[3:6, 9:12] = -R*dt
    Fx[3:6, 15:18] = Idt

    # Third row
    Fx[6:9, 6:9] = Rotation.as_matrix( Rotation.from_rotvec(((w_m - w_b)*dt).flatten()) ).T
    Fx[6:9, 12:15] = -Idt
    # Fx[9:12, 9:12] = Fx[12:15, 12:15] = Fx[15:18, 15:18] = np.eye(3)

    V_i = accelerometer_noise_density**2 * dt * Idt
    theta_i = gyroscope_noise_density**2 * dt * Idt
    a_i = accelerometer_random_walk**2 * Idt
    ohm_i = gyroscope_random_walk**2 * Idt

    Qi = np.zeros((12, 12))
    Qi[0:3, 0:3] = V_i
    Qi[3:6, 3:6] = theta_i
    Qi[6:9, 6:9] = a_i
    Qi[9:12, 9:12] = ohm_i

    # print('Qi: ', Qi)
    # exit()

    # print('Fx: ', Fx.shape)
    # print('Vi: ', Vi)
    # print('thetai: ', thetai)
    # print('ai: ', ai)
    # print('omega_i: ', omega_i)


    # exit()
    Fi = np.zeros((18, 12))
    Fi[3:6, 0:3] = Fi[6:9, 3:6] = Fi[12:15, 9:12] = Fi[9:12, 6:9] = np.eye(3)

    new_cov_mat = (Fx @ error_state_covariance @ Fx.T) + (Fi @ Qi @ Fi.T)

    # YOUR CODE HERE

    # return an 18x18 covariance matrix
    return new_cov_mat


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R = Rotation.as_matrix(q)  

    P_c = (R.T @ (Pw - p)).flatten()
    # print('P_c', P_c.shape)
    # exit()
    Z_c = P_c[2]
    pred_uv = np.array([P_c[0]/Z_c, P_c[1]/Z_c]).reshape(-1, 1)
    innovation = uv - pred_uv
    # print(innovation.shape)

    if np.linalg.norm(innovation) < error_threshold:

        dz_dPc = (1/Z_c) * np.array([1, 0, -pred_uv[0,0], 0, 1, -pred_uv[1,0]]).reshape(2, 3)
        dP_dp = np.array([0, -P_c[2], P_c[1], P_c[2], 0, -P_c[0], -P_c[1], P_c[0], 0]).reshape(3, 3)
        dP_del_p = -(R.T)

        # dz_dtheta = dz_dPc  @ dP_dp 
        # dz_dp = dz_dPc @ dP_del_p

        H_t = np.zeros((2, 18))
        H_t[:, 0:3] = dz_dPc @ dP_del_p
        H_t[:, 6:9] = dz_dPc @ dP_dp

        # print('H_t: ', H_t.shape)
        # print('H_t: ', H_t)
        # exit()

        Kalman_Gain = error_state_covariance @ H_t.T @ np.linalg.inv((H_t @ error_state_covariance @ H_t.T) + Q)
        # print('Kalman Gain: ', Kalman_Gain.shape)
        # print('Kalman Gain: ', Kalman_Gain)
        # exit()
        del_x = (Kalman_Gain @ innovation)
        # print(delx.shape)
        # exit()

        delp = del_x[0:3]
        delv = del_x[3:6]
        dela_b = del_x[9:12]
        delw_b = del_x[12:15]
        delg = del_x[15:18]

        p = p + delp
        v = v + delv
        q = q * Rotation.from_rotvec((del_x[6:9]).flatten())
        a_b = a_b + dela_b
        w_b = w_b + delw_b
        g = g + delg

        Iden = np.eye(18)
        error_state_covariance = (Iden-Kalman_Gain @ H_t) @ error_state_covariance @ (Iden-Kalman_Gain @ H_t).T + (Kalman_Gain @ Q @ Kalman_Gain.T)

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
