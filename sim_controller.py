#!/usr/bin/env python

'''
CONTROLLER for simulation setup
data collection with airsim and unreal for:
simple_flight uav
zed mini stereo camera
    rgb left, rgb right, depth


COORDINATE SYSTEM:
All AirSim API uses NED coordinate system, i.e., +X is North, +Y is East and +Z is Down. All units are in SI system. Please note that this is different from coordinate system used internally by Unreal Engine. In Unreal Engine, +Z is up instead of down and length unit is in centimeters instead of meters. AirSim APIs takes care of the appropriate conversions. The starting point of the vehicle is always coordinates (0, 0, 0) in NED system. Thus when converting from Unreal coordinates to NED, we first subtract the starting offset and then scale by 100 for cm to m conversion. The vehicle is spawned in Unreal environment where the Player Start component is placed. There is a setting called OriginGeopoint in settings.json which assigns geographic longitude, longitude and altitude to the Player Start component.
'''

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
from pid import PID

#----------------- Lab Sim setup configuration --------------------------------
GATE_BASE_NAME = "BP_AirLab2m1Gate_"
DATASET_BASENAME = "X4Gates_Circle_"
DATASET_BASEPATH = "/home/kristoffer/dev/dataset"


'''
controller, that generates training data for imitation learning with AirSim in the Skejby Lab environment
sim_controller.move() contains main functionality
'''
class sim_controller:

    def __init__(self):

        ''' 
        INIT VALUES 
        '''

        # frame rate
        self.framerate = 30 # fps
        self.timestep = 1./self.framerate # hz
        self.roundtime = 15 # seconds

        # append run number to base name
        self.DATASET_NAME = DATASET_BASENAME + "0"
        self.DATASET_PATH = DATASET_BASEPATH + "/" + self.DATASET_NAME
        self.DATASET_PATH_LEFT = self.DATASET_PATH + "/image_left"
        self.DATASET_PATH_RIGHT = self.DATASET_PATH + "/image_right"
        self.DATASET_PATH_DEPTH = self.DATASET_PATH + "/image_depth"


        '''
        CONNECTION TO AIRSIM
        '''
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # shortcut
        self.c = curses.initscr()
        curses.noecho()
        curses.cbreak()


        '''
        CREATE DATASET FOLDERS
        '''
        # root path of the data set to be created
        self.createFolder(self.DATASET_PATH)
        # left images will be saved here
        self.createFolder(self.DATASET_PATH_LEFT)
        # right images will be saved here
        self.createFolder(self.DATASET_PATH_RIGHT)
        # depth images will be saved here
        self.createFolder(self.DATASET_PATH_DEPTH)

    # get current position of UAV from simulation
    # returns [x, y, z, qw, qx, qy, qz]
    def getPositionUAV(self):
        dronePose = self.client.simGetVehiclePose()
        retVal = np.array([dronePose.position.x_val, dronePose.position.y_val, dronePose.position.z_val, dronePose.orientation.w_val, dronePose.orientation.x_val, dronePose.orientation.y_val, dronePose.orientation.z_val])
        return retVal
    # get current position of UAV from simulation
    # returns airsim.Pose()
    def getPositionAirsimUAV(self):
        return self.client.simGetVehiclePose()

    def getState(self):
        pos = self.getPositionAirsimUAV()
        _, _, yaw = airsim.to_eularian_angles(pos.orientation)
        return [pos.position.x_val, pos.position.y_val, pos.position.z_val, yaw]

    # get position of gate with index idx
    # returns [x, y, z, qw, qx, qy, qz]
    def getPositionGate(self, idx):
        gatePose = self.client.simGetObjectPose(GATE_BASE_NAME + str(idx))
        retVal = np.array([gatePose.position.x_val, gatePose.position.y_val, gatePose.position.z_val, gatePose.orientation.w_val, gatePose.orientation.x_val, gatePose.orientation.y_val, gatePose.orientation.z_val])
        return retVal
    # get position of gate with index idx
    # returns airsim.Pose()
    def getPositionAirsimGate(self, idx):
        return self.client.simGetObjectPose(GATE_BASE_NAME + str(idx))

    # set position of gate to pos with yaw as direction to face
    # idx: gate index
    # pos: new position of gate
    # yaw: direction, rad/deg?
    def setPositionGate(self, idx, pos, yaw):
        pose = airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(0, 0, -yaw))
        if idx < 1 or idx > 4:
            print("setPositionGate: error gate idx not found.")
            return
        self.client.simSetObjectPose(GATE_BASE_NAME + str(idx), pose, True)

    # capture and save three images, left and right rgb, depth
    def captureAndSaveImages(self):
        res = self.client.simGetImages(
            [
                airsim.ImageRequest("front_left", airsim.ImageType.Scene),
                airsim.ImageRequest("front_right", airsim.ImageType.Scene),
                airsim.ImageRequest("depth_cam", airsim.ImageType.DepthPlanner, True)
            ]
        )
        left = res[0]
        right = res[1]
        depth = res[2]

        # save left image
        airsim.write_file(self.DATASET_PATH_LEFT + "/testimg.png", left.image_data_uint8)
        # save right image
        airsim.write_file(self.DATASET_PATH_RIGHT + "/testimg.png", right.image_data_uint8)
        # save depth as portable float map
        airsim.write_pfm(self.DATASET_PATH_DEPTH + "/testimg.pfm", airsim.get_pfm_array(depth))

    # gets current position of gates and generates a trajectory through those points
    # returns maveric.Waypoint objects, trajectory as list of parameters for polynomials (see convertTrajectoryToWaypoints())
    def generateTrajectoryFromCurrentGatePositions(self, timestep=1):
        
        waypoints = []

        # uav current position is first waypoint
        uavpos = self.getPositionUAV()
        orientation = self.getPositionAirsimUAV().orientation
        # convert to xyz and yaw
        _, _, yaw = airsim.to_eularian_angles(orientation)
        uavwp = [uavpos[0], uavpos[1], uavpos[2], yaw]
        # add waypoint
        waypoints.append(uavwp)


        # get current gate positions
        for i in range(1,5):
            # get gate position
            gp = self.getPositionGate(i)
            orientation = self.getPositionAirsimGate(i).orientation
            # convert to xyz and yaw
            _, _, yaw = airsim.to_eularian_angles(orientation)
            wp = [gp[0], gp[1], gp[2], yaw]
            # add waypoint
            waypoints.append(wp)

        # add uavwp again - starting point as endpoint
        waypoints.append(uavwp)

        # call maveric to get trajectory
        return maveric.planner(waypoints, timestep=timestep)

    # convert waypoints, trajectory from generateTrajectoryFromCurrentGatePositions() to airsim.Vector3r list and [x,y,z,yaw,timestamp]
    def convertTrajectoryToWaypoints(self, waypoints, trajectory, evaltime=10):
        out = []
        outComplete = []

        for i in range(len(trajectory)):

            t = np.linspace(waypoints[i].time, waypoints[i + 1].time, (int(waypoints[i + 1].time) - int(waypoints[i].time)) * evaltime)
            x_path = (trajectory[i][0] * t ** 4 + trajectory[i][1] * t ** 3 + trajectory[i][2] * t ** 2 + trajectory[i][3] * t + trajectory[i][4])
            y_path = (trajectory[i][5] * t ** 4 + trajectory[i][6] * t ** 3 + trajectory[i][7] * t ** 2 + trajectory[i][8] * t + trajectory[i][9])
            z_path = (trajectory[i][10] * t ** 4 + trajectory[i][11] * t ** 3 + trajectory[i][12] * t ** 2 + trajectory[i][13] * t + trajectory[i][14])
            yaw_path = (trajectory[i][15] * t ** 2 + trajectory[i][16] * t + trajectory[i][17])

            for j in range(len(t)):
                out.append(airsim.Vector3r(x_path[j], y_path[j], z_path[j]))
                outComplete.append([x_path[j], y_path[j], z_path[j], yaw_path[j], t[j]])

        return out, outComplete

    # print pose list 
    # pose: [x, y, z, qw, qx, qy, qz]
    def printPose(self, pose):
        print(f"x: {pose[0]} y: {pose[1]} z: {pose[2]} qw: {pose[3]} qx: {pose[4]} qy: {pose[5]} qz: {pose[6]} ")

    # get the current uav position and print it
    def getAndPrintCurrentPosition(self):
        cp = self.getPositionUAV()
        print("current Position:")
        self.printPose(cp)
        return cp

    # create folder at path with name
    def createFolder(self, path, name=""):
        folder = ""
        if name == "":
            folder = path
        else:
            folder = os.path.join(path, name)

        if not os.path.isdir(folder):
            print(f"created folder '{folder}'")
            os.mkdir(folder)

    # reset simulation environment
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

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
        ctrl = PID()

        firstGoal = self.getState()
        # firstGoal[3] = 180
        ctrl.setGoal(firstGoal)

        # get trajectory
        timed_waypoints, trajectory = self.generateTrajectoryFromCurrentGatePositions(timestep=1)

        # print("follow generated path from gates")

        path, pathComplete = self.convertTrajectoryToWaypoints(timed_waypoints, trajectory, evaltime=self.roundtime)

        # show trajectory
        self.client.simPlotPoints(path, color_rgba=[1.0, 0.0, 0.0, .2], size = 10.0, duration = -1.0, is_persistent = True)

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





    '''
    close all opened files here
    '''
    def close(self):
        curses.nocbreak()
        # curses.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        self.client.simFlushPersistentMarkers()
        return



import contextlib
with contextlib.closing(sim_controller()) as sc:
    sc.gateMission()

