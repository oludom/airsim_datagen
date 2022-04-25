#!/usr/bin/env python

'''

simulation client for AirSimController.py 
this runs the main loop and holds the settings for the simulation. 


'''

from AirSimController import AirSimController

import airsim
import numpy as np
import pprint
import curses

import os
import time
from math import *
import time

import cv2

from numpy.random import default_rng

# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
from VelocityPID import VelocityPID

class SimClient(AirSimController):

    def __init__(self):
        
        # do custom setup here

        # shortcut
        self.c = curses.initscr()
        curses.noecho()
        curses.cbreak()

        # init super class (AirSimController)
        super().__init__()



    '''
    close all opened files here
    '''
    def close(self):
        curses.nocbreak()
        # curses.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        self.client.simFlushPersistentMarkers()
        

    # test following a trajectory with setSimVehiclePose
    def trajectory_test(self):
        self.reset()

        self.client.takeoffAsync().join()

        # get trajectory
        timed_waypoints, trajectory = self.generateTrajectoryFromCurrentGatePositions()

        print("follow generated path")

        path, pathComplete = self.convertTrajectoryToWaypoints(timed_waypoints, trajectory)
        # print(path)
        # moveOnPathAsync(self, path, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        # self.client.moveOnPathAsync(path, 2, timeout_sec=3e+38, yaw_mode=airsim.YawMode(False, 100.0)).join()



        # draw trajectory
        # simPlotPoints(self, points, color_rgba=[1.0, 0.0, 0.0, 1.0], size = 10.0, duration = -1.0, is_persistent = False):
        self.client.simPlotPoints(path, duration=60)

        # dronePose = self.client.simGetVehiclePose()
        # qyaw = airsim.to_quaternion(0, 0, -pathComplete[0][3])
        # dronePose.position.x_val = pathComplete[0][0]
        # dronePose.position.y_val = pathComplete[0][1]
        # dronePose.position.z_val = pathComplete[0][2]
        # dronePose.orientation.w_val = qyaw.w_val
        # dronePose.orientation.x_val = qyaw.x_val
        # dronePose.orientation.y_val = qyaw.y_val
        # dronePose.orientation.z_val = qyaw.z_val
        # self.client.simSetVehiclePose(dronePose, True)

        for i, point in enumerate(pathComplete):
            dronePose = self.client.simGetVehiclePose()
            qyaw = airsim.to_quaternion(0, 0, -point[3])
            dronePose.position.x_val = point[0]
            dronePose.position.y_val = point[1]
            dronePose.position.z_val = point[2]
            dronePose.orientation.w_val = qyaw.w_val
            dronePose.orientation.x_val = qyaw.x_val
            dronePose.orientation.y_val = qyaw.y_val
            dronePose.orientation.z_val = qyaw.z_val
            self.client.simSetVehiclePose(dronePose, True)
            p = [point[0], point[1], point[2], qyaw.w_val, qyaw.x_val, qyaw.y_val, qyaw.z_val]
            self.printPose(p)
            # self.client.simSetObjectPose(pose, True)
            time.sleep(0.1)

    # move UAV around a bit to show that api control is active and working
    def move(self):

        
        self.client.takeoffAsync().join()


        # capture test images
        self.captureAndSaveImages()

        cp = self.getAndPrintCurrentPosition()

        # get gate position and print it
        for i in range(1,5):
            gp = self.getPositionGate(i)
            print(f"gate {i} Position:")
            self.printPose(gp)


        np = cp
        np[0] = np[0] + 2

        self.client.moveToPositionAsync(np[0], np[1], -.5, 2).join()

        cp = self.getAndPrintCurrentPosition()

        time.sleep(2)

        print("goto: gate 1 position")
        self.client.moveToPositionAsync(gp[0], gp[1], -.5, 2).join()

        cp = self.getAndPrintCurrentPosition()

        
    def gateMission(self):

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        # init pid controller for velocity control

        pp = 4.
        dd = 0.01
        ii = 0.5
        
        Kp = np.array([pp, pp, .45])
        Ki = np.array([ii, ii, 0.0])
        Kd = np.array([dd, dd, 0.0])
        yaw_gain = 0.0

        distance_threshold = 0.01
        angle_threshold = 0.1

        ctrl = VelocityPID(Kp, Ki, Kd, yaw_gain, distance_threshold, angle_threshold)

        firstGoal = self.getState()
        # firstGoal[3] = 180
        ctrl.setGoal(firstGoal)

        # get trajectory
        timed_waypoints, trajectory = self.generateTrajectoryFromCurrentGatePositions(timestep=1)

        # print("follow generated path from gates")

        path, pathComplete = self.convertTrajectoryToWaypoints(timed_waypoints, trajectory, evaltime=self.roundtime)

        # show trajectory
        # self.client.simPlotPoints(path, color_rgba=[1.0, 0.0, 0.0, .2], size = 10.0, duration = -1.0, is_persistent = True)

        # for wp in pathComplete:

        lastTs = time.time()

        cwpindex = 0
        timePerWP = float(self.roundtime) / len(pathComplete) 

        lastWaypointTs = time.time()

        # controll loop
        while True:

            # get and plot current waypoint
            wp = pathComplete[cwpindex]
            # convert radians to degrees
            # wp[3] = 0
            self.client.simPlotPoints([airsim.Vector3r(wp[0], wp[1], wp[2])], color_rgba=[0.0, 0.0, 1.0, 1.0], size = 10.0, duration = self.timestep, is_persistent = False)

            # get current time and time delta
            tn = time.time()
            # time delta
            dt = tn - lastTs

            # wait remaining time until time step has passed
            remainingTime = (self.timestep) - dt
            if remainingTime > 0:
                time.sleep(remainingTime)

            # get current time again
            tn = time.time()
            # new time delta
            dt = tn - lastTs

            # calculate actual frequency
            hz = 1./float(dt)

            self.c.addstr(0,0, "following generated path from gates...")
            self.c.addstr(2,0, f"frame rate: {hz}")

            # get current state
            cstate = self.getState()
            # inform pid controller about state
            ctrl.setState(cstate)


            ctrl.setGoal(wp[:4])
            # update pid controller
            ctrl.update()
            self.c.addstr(3,0, ctrl.errorOutput)
            # get current pid output´
            vel, yaw = ctrl.getVelocityYaw()

            '''
            Args:
                vx (float): desired velocity in world (NED) X axis
                vy (float): desired velocity in world (NED) Y axis
                vz (float): desired velocity in world (NED) Z axis
                duration (float): Desired amount of time (seconds), to send this command for
                drivetrain (DrivetrainType, optional):
                yaw_mode (YawMode, optional):
                vehicle_name (str, optional): Name of the multirotor to send this command to
            '''
            self.client.moveByVelocityAsync(float(vel[0]), float(vel[1]), float(vel[2]), duration=float(self.timestep), yaw_mode=airsim.YawMode(False, degrees(-yaw)))

            # debug output
            self.c.addstr(5,0,f"yaw state: {cstate[3]}")
            self.c.addstr(6,0,f"yaw goal: {wp[3]}")
            self.c.addstr(7,0,f"7")
            self.c.addstr(8,0,f"8")

            self.c.refresh()
            lastTs = tn

            # increase current waypoint index if time per waypoint passed and if there are more waypoints available in path
            if tn - lastWaypointTs > timePerWP and len(pathComplete) > (cwpindex+1):
                cwpindex = cwpindex + 1
                lastWaypointTs = tn




import contextlib
with contextlib.closing(SimClient()) as sc:
    sc.gateMission()