import math

import numpy as np
import math as m


# calculate magnitude of vector
def magnitude(vec):
    return np.sqrt(vec.dot(vec))


# theta: angle in radian
def Rx(theta: float):
    return np.array([[1, 0, 0],
                     [0, m.cos(theta), m.sin(theta)],
                     [0, -m.sin(theta), m.cos(theta)]])


# theta: angle in radian
def RxT(theta: float):
    return Rx(theta).T


# theta: angle in radian
def Ry(theta: float):
    return np.array([[m.cos(theta), 0, -m.sin(theta)],
                     [0, 1, 0],
                     [m.sin(theta), 0, m.cos(theta)]])


# theta: angle in radian
def RyT(theta: float):
    return Ry(theta).T


# theta: angle in radian
def Rz(theta: float):
    return np.array([[m.cos(theta), m.sin(theta), 0],
                     [-m.sin(theta), m.cos(theta), 0],
                     [0, 0, 1]])


# theta: angle in radian
def RzT(theta: float):
    return Rz(theta).T


# convert vector from world to body frame
# v: vector to transform in world frame
# b: position of body in world frame / translation offset
# by: body yaw in radian
def vector_world_to_body(v: np.ndarray, b: np.ndarray, by: float) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return Rz(by) @ (v - b)


# convert vector from body to world frame
# v: vector to transform in body frame
# b: position of body in world frame / translation offset
# by: body yaw in radian
def vector_body_to_world(v: np.ndarray, b: np.ndarray, by: float) -> np.ndarray:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return (RzT(by) @ v) + b


# helper method for converting getOrientation to roll/pitch/yaw
# https:#en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

def to_eularian_angles(w, x, y, z):
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)

    return pitch, roll, yaw


def to_quaternion(pitch, roll, yaw):
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    w = t0 * t2 * t4 + t1 * t3 * t5  # w
    x = t0 * t3 * t4 - t1 * t2 * t5  # x
    y = t0 * t2 * t5 + t1 * t3 * t4  # y
    z = t1 * t2 * t4 - t0 * t3 * t5  # z
    return w, x, y, z
