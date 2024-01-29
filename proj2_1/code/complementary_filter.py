# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def normalize(q):
    """
    Normalizes a vector

    :param q: vector to normalize
    :return: normalized vector
    """
    q = q/np.linalg.norm(q)
    return q


def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    # TODO Your code here - replace the return value with one you compute
    Rotation_k = initial_rotation * Rotation.from_rotvec(angular_velocity * dt)
    ex = np.array([1, 0, 0])
    g_dash = normalize((Rotation_k.as_matrix() @ linear_acceleration))
    inner_dot_product = g_dash @ ex
    angle = np.arccos(inner_dot_product)

    inner_cross_product = np.cross(g_dash, ex)
    omega_correction = normalize(inner_cross_product)

    norm_accel = norm(linear_acceleration)/9.81
    em = np.abs(norm_accel - 1)

    alpha = np.where(em >= 0.2, 0, np.where(em <= 0.1, 1, -10*(em-0.1) + 1))

    a1 = (1-alpha) * np.array([0, 0, 0, 1])
    quat_correction = Rotation.from_rotvec(omega_correction * angle)
    a2 = alpha * quat_correction.as_quat()        
    quat_correction_dash = a1 + a2
    quat_correction_dash = Rotation.from_quat(quat_correction_dash)
    
    quat1 = Rotation.from_quat(quat_correction_dash.as_quat()).as_matrix()
    quat2 = Rotation.from_quat(Rotation_k.as_quat()).as_matrix()
    quat = quat1 @ quat2
    quaternion_k = Rotation.from_matrix(quat).as_quat()
    new_rotation_estimate = Rotation.from_quat(quaternion_k)
    
    return new_rotation_estimate
