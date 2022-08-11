#!/usr/bin/env python

import time
import numpy as np
from math import *

'''
PID controller based on 
https://vazgriz.com/621/pid-controllers/
'''


class PIDController:
    # pg: p-gain
    # ig: i-gain
    # dg: d-gain
    def __init__(self, pg, ig, dg):
        self.proportionalGain = pg
        self.integralGain = ig
        self.derivativeGain = dg

        self.valueLast = 0
        self.integrationStored = 0

        self.errorOutput = ""

    # use for linear values e.g. x, y, z
    def update(self, dt: float, currentValue: float, targetValue: float):
        error = targetValue - currentValue

        P = self.proportionalGain * error

        self.integrationStored = self.integrationStored + (error * dt)
        I = self.integralGain * self.integrationStored
        # I = max(min(I, 5), -5)

        valueRateOfChange = (currentValue - self.valueLast) / dt
        self.valueLast = currentValue

        D = self.derivativeGain * -valueRateOfChange

        self.errorOutput = f"Pe: {error}, De: {-valueRateOfChange}"

        return P + I + D

    # all angles in degree
    def angleDifference(self, a: float, b: float):
        return (a - b + 540) % 360 - 180

    # use for angle values e.g. yaw
    def updateAngle(self, dt: float, currentAngle: float, targetAngle: float):
        error = self.angleDifference(targetAngle, currentAngle)
        print(f"target: {targetAngle}, current: {currentAngle}, error: {error}")

        P = self.proportionalGain * error

        self.integrationStored = self.integrationStored + (error * dt)
        I = self.integralGain * self.integrationStored
        I = max(min(I, 2), -2)

        valueRateOfChange = self.angleDifference(currentAngle, self.valueLast) / dt
        self.valueLast = currentAngle

        D = self.derivativeGain * -valueRateOfChange

        self.errorOutput = f"Pe: {error}, De: {-valueRateOfChange}"

        return max(min(P + I + D, self.yaw_limit), -self.yaw_limit)


'''
velocity pid controller
outputs a velocity command in world frame, 
based on input x, y, z, and yaw in world frame

'''


class VelocityPID:
    '''
    pid controller
    kp: np.array 3, p gain for x, y, z
    ki: np.array 3, i gain for x, y, z
    kd: np.array 3, d gain for x, y, z
    yg: np.array 3, yaw gains, p, i, d
    dthresh: float, distance threshold
    athresh: float, angle threshold

    '''

    def __init__(self, kp=np.array([0., 0., 0.]), ki=np.array([0., 0., 0.]), kd=np.array([0., 0., 0.]), yg=0.,
                 dthresh=.1, athresh=.1):
        # in degrees, max requested change
        self.yaw_limit = 10

        # state
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0

        # previous
        self.previous_distance_error = np.array([0, 0, 0])
        self.previous_angle_error = np.array([0, 0, 0])
        self.previous_integral_error = np.array([0, 0, 0])

        # Default state
        self.x_goal = 0
        self.y_goal = 0
        self.z_goal = 0
        self.yaw_goal = 0

        # gains
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.yaw_gain = yg

        # output values
        self.velocity_out = np.array([0, 0, 0])
        self.yaw_out = 0

        # error publisher
        self.errorOutput = ""
        self.errorOutput1 = ""
        self.errorOutput2 = ""
        self.errorOutput3 = ""

        # PID controller objects
        self.cx = PIDController(kp[0], ki[0], kd[0])
        self.cy = PIDController(kp[1], ki[1], kd[1])
        self.cz = PIDController(kp[2], ki[2], kd[2])
        self.cyaw = PIDController(yg[0], yg[1], yg[2])

    # [x, y, z, yaw]
    def setState(self, state):
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.yaw = state[3]

    # [x, y, z, yaw]
    def setGoal(self, state):
        self.x_goal = state[0]
        self.y_goal = state[1]
        self.z_goal = state[2]
        self.yaw_goal = state[3]

    # returns output of pid controller
    def getVelocityYaw(self):
        return (self.velocity_out, self.yaw_out)

    # df: time delta since last update
    def update(self, dt):
        # Retrieve the UAV's state
        x = self.x
        y = self.y
        z = self.z
        yaw = self.yaw

        goal_x = self.x_goal
        goal_y = self.y_goal
        goal_z = self.z_goal

        # set current output values
        self.velocity_out = np.array(
            [self.cx.update(dt, x, goal_x), self.cy.update(dt, y, goal_y), self.cz.update(dt, z, goal_z)])
        self.yaw_out = self.cyaw.update(dt, yaw, self.yaw_goal)
        self.yaw_out = max(min(self.yaw_out, self.yaw_limit), -self.yaw_limit)

        # error output
        self.errorOutput = ""
        self.errorOutput1 = f"x: {self.cx.errorOutput}"
        self.errorOutput2 = f"y: {self.cy.errorOutput}"
        self.errorOutput3 = f"z: {self.cz.errorOutput}"
