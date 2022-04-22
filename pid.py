#!/usr/bin/env python

import time
import numpy as np
from math import *

'''
custom pid controller

'''


class PID:

    def __init__(self):


        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0

        
        self.previous_time = time.time()
        self.previous_distance_error = np.array([0, 0, 0])
        self.previous_angle_error = np.array([0, 0, 0])
        self.integral = np.array([0, 0, 0])

        # Default parameters: CHANGE HERE
        self.x_goal = 0
        self.y_goal = 0
        self.z_goal = 0
        self.yaw_goal = 0
        
        self.Kp = np.array([0.1, 0.1, 0.1])
        self.Ki = np.array([0.2, 0.2, 0.2])
        self.Kd = np.array([0.05, 0.05, 0.03])
        self.yaw_gain = 0.2

        self.distance_threshold = 0.01
        self.angle_threshold = 0.1

        # output values
        self.velocity = np.array([0,0,0])
        self.yaw_out = 0

        # error publisher
        self.errorOutput = ""



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
        return (self.velocity, self.yaw_out)


    def update(self):

        previous_distance_error = self.previous_distance_error
        distance_integral = self.integral

        # Retrieve the UAV's state
        x = self.x
        y = self.y
        z = self.z
        yaw = self.yaw

        Rbv = np.array([[cos(yaw), sin(yaw), 0], [-1*sin(yaw), cos(yaw), 0], [0, 0, 1]])    # roll = pitch = 0

        goal_x = self.x_goal
        goal_y = self.y_goal
        goal_z = self.z_goal

        # Time lapse estimation

        # Local machine time (for simulation: ok, for real drone implementation: no)
        time_now = time.time()
        dt = time_now - self.previous_time
        self.previous_time = time_now
        # From message time stamp (more realistic)

        # Estimate error in World (Initial) Frame

        distance_error = np.array([goal_x - x, goal_y - y, goal_z - z])
        distance_integral = distance_integral + distance_error * dt
        distance_derivative = (distance_error - previous_distance_error)/dt

        # Store
        previous_distance_error = distance_error


        vel_world = self.Kp*distance_error + self.Ki*distance_integral + self.Kd*distance_derivative
        yaw_rate_world = (self.yaw_goal - yaw)*self.yaw_gain

        distance = sqrt((goal_z - z)**2 + (goal_y - y)**2 + (goal_x - x)**2)
        angle_error = float(abs(self.yaw_goal - yaw))

        # disply and publish distance and angle error
        # print "distance = ", distance, "    , angle error = ", angle_error, "   , dt = ", dt
        # print(f"distance: {distance}, angle error: {angle_error}")
        self.errorOutput = f"distance: {distance}, angle error: {angle_error}"

        

        if distance >= self.distance_threshold:

            # Implement PD controller

            # calculate into body frame
            # vel_body = Rbv.dot(np.transpose(vel_world))
            # self.velocity = vel_body
            self.velocity = vel_world


        # else:
        #     # Stopping our robot after the movement is over.
        #     vel_msg.linear.x = 0
        #     self.velocity_publisher.publish(vel_msg)
        #     self.integral = np.array([0, 0, 0])

        if angle_error >= self.angle_threshold:
        #     yaw_rate_body = yaw_rate_world * 1  #roll = pitch = 0
        #     vel_msg.angular.z = yaw_rate_body
        #     self.velocity_publisher.publish(vel_msg)
            self.yaw_out = yaw_rate_world
        else: 
            self.yaw_out = 0



# def quaternion_to_euler(x, y, z, w):

#         import math
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         X = math.degrees(math.atan2(t0, t1))

#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         Y = math.degrees(math.asin(t2))

#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         Z = math.degrees(math.atan2(t3, t4))

#         X = math.atan2(t0, t1)
#         Y = math.asin(t2)
#         Z = math.atan2(t3, t4)

#         return X, Y, Z