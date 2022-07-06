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

import os
import time
from math import *
import time
import json
from copy import deepcopy

import cv2

import MAVeric.trajectory_planner as maveric


'''

'''
class AirSimController:

    def __init__(self, raceTrackName, createDataset=True):

        ''' 
        INIT VALUES 
        '''

        # configuration file
        self.configFile = open('config.json', "r")

        self.config = {}
        self.loadConfig(self.configFile)

        # append run number to base name
        self.DATASET_NAME = self.config.dataset_basename
        self.DATASET_PATH_SUPER = self.config.dataset_basepath + "/" + self.DATASET_NAME
        self.DATASET_PATH = self.DATASET_PATH_SUPER + "/" + raceTrackName
        self.DATASET_PATH_FILE = self.DATASET_PATH + "/data.json"
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

        self.createDataset = createDataset

        if createDataset:

            '''
            CREATE DATASET FOLDERS
            '''
            # root path of the data set to be created
            self.createFolder(self.DATASET_PATH_SUPER)
            self.createFolder(self.DATASET_PATH)
            # left images will be saved here
            self.createFolder(self.DATASET_PATH_LEFT)
            # right images will be saved here
            self.createFolder(self.DATASET_PATH_RIGHT)
            # depth images will be saved here
            self.createFolder(self.DATASET_PATH_DEPTH)

            '''
            CREATE DATASET OUTPUT FILE
            '''
            self.outputFile = open(self.DATASET_PATH_FILE, "w")
            print('{\n"data": [', file=self.outputFile)

        # imu cache for faster data collection
        self.imuCache = []

    # save config file to data set folder
    # gates: np.array of shape [x, y, z, yaw]
    # data: dict - key value
    def saveConfigToDataset(self, gates, data={}):
        if self.createDataset:
            dconfigfile = open(self.DATASET_PATH + "/config.json", "w")
            self.configFile.seek(0)
            dconfig = json.load(self.configFile)
            dconfig['gates']['poses'] = gates.tolist()
            dconfig['gates']['number_configurations'] = 1

            for key, value in data.items():
                dconfig[key] = value

            json.dump(dconfig, dconfigfile, indent="  ")
            dconfigfile.close()


    def loadConfig(self, configFile):
        class ConfigLoader:
            def __init__(self, **data):
                self.__dict__.update(data)
        data = json.load(configFile)
        self.config = ConfigLoader(**data)

    '''
    this function should be called at end of lifetime of this object (see contextlib)
    close all opened files here
    and reset simulation
    '''
    def close(self):
        self.configFile.close()
        if self.createDataset:
            # ending of json file
            self.outputFile.flush()
            # remove last comma
            l = self.outputFile.tell()
            self.outputFile.seek(l-2)
            print(']\n}', file=self.outputFile)
            self.outputFile.close()


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

    # get current state of UAV
    # returns [x, y, z, yaw]
    def getState(self):
        pos = self.getPositionAirsimUAV()
        _, _, yaw = airsim.to_eularian_angles(pos.orientation)
        return [pos.position.x_val, pos.position.y_val, pos.position.z_val, yaw]

    # get position of gate with index idx
    # returns [x, y, z, qw, qx, qy, qz]
    def getPositionGate(self, idx):
        gatePose = self.client.simGetObjectPose(self.config.gate_basename + str(idx))
        retVal = np.array([gatePose.position.x_val, gatePose.position.y_val, gatePose.position.z_val, gatePose.orientation.w_val, gatePose.orientation.x_val, gatePose.orientation.y_val, gatePose.orientation.z_val])
        return retVal
    # get position of gate with index idx
    # returns airsim.Pose()
    def getPositionAirsimGate(self, idx):
        return self.client.simGetObjectPose(self.config.gate_basename + str(idx))

    # set position of gate to pos with yaw as direction to face
    # idx: gate index
    # pos: new position of gate
    # yaw: direction, deg
    def setPositionGate(self, idx, pos, yaw=None):
        if yaw == None:  # assume pos[3] is yaw
            yaw = pos[3]
        pose = airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]), airsim.to_quaternion(0, 0, radians(yaw)))
        if idx < 1 or idx > 4:
            print("setPositionGate: error gate idx not found.")
            return
        self.client.simSetObjectPose(self.config.gate_basename + str(idx), pose, True)


    # ImuData: type ImuData (airsim sensor message)
    # returns time stamp, dict of ImuData
    def convertImuToDict(self, ImuData):
        imu = {}
        imu['angular_velocity'] = {
            "x": ImuData.angular_velocity.x_val,
            "y": ImuData.angular_velocity.y_val,
            "z": ImuData.angular_velocity.z_val
        }
        imu['linear_acceleration'] = {
            "x": ImuData.linear_acceleration.x_val,
            "y": ImuData.linear_acceleration.y_val,
            "z": ImuData.linear_acceleration.z_val
        }
        imu['orientation'] = {
            "x": ImuData.orientation.x_val,
            "y": ImuData.orientation.y_val,
            "z": ImuData.orientation.z_val,
            "w": ImuData.orientation.w_val
        }
        return (ImuData.time_stamp, imu)


    # capture and save three images, left and right rgb, depth
    # wpidx: index of current waypoint, that is targeted by controller
    # idx: image index, used for naming the images, should count up to prevent overwriting existing images
    def captureAndSaveImages(self, wpidx, idx=0):

        # current frame name
        cfname = "image" + str(idx)

        # AirSim API rarely returns empty image data
        # 'and True' emulates a do while loop
        loopcount = 0
        while(self.createDataset and True):

            # get images from AirSim API
            res = self.client.simGetImages(
                [
                    airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                    # airsim.ImageRequest("front_right", airsim.ImageType.Scene),
                    airsim.ImageRequest("depth_cam", airsim.ImageType.DepthPlanar, True)
                ]
            )
            left = res[0]
            # right = res[1]
            depth = res[1]


            # save left image
            # airsim.write_file(self.DATASET_PATH_LEFT + f"/{cfname}.png", left.image_data_uint8)
            img1d = np.fromstring(left.image_data_uint8, dtype=np.uint8) # get numpy array
            img_rgb = img1d.reshape(left.height, left.width, 3) # reshape array to 3 channel image array H X W X 3
        
            # check if image contains data, repeat request if empty
            if img_rgb.size:
                break  # end of do while loop
            else:
                loopcount += 1
                print("airsim returned empty image." + str(loopcount))

        if self.createDataset:
            cv2.imwrite(self.DATASET_PATH_LEFT + f"/{cfname}.png", img_rgb) # write to png
            # save right image
            # airsim.write_file(self.DATASET_PATH_RIGHT + f"/{cfname}.png", right.image_data_uint8)
            # save depth as portable float map
            airsim.write_pfm(self.DATASET_PATH_DEPTH + f"/{cfname}.pfm", airsim.get_pfm_array(depth))

        ts = self.captureIMU()
        imu = self.imuCache
        # reset imu cache
        self.imuCache = []

        # get UAV pose
        pos = list(self.getPositionUAV())
        
        # write entry to output file
        entry = {
            "image_name": cfname,
            "time_stamp": ts,
            "waypoint_index": wpidx,
            "pose": pos,
            "imu": imu
        }
        if self.createDataset:
            print(f"{json.dumps(entry, indent=1)},", file=self.outputFile)


    def captureIMU(self):
        # get imu data
        imuData = self.client.getImuData()
        ts, imu = self.convertImuToDict(imuData)
        self.imuCache.append([ts, imu])
        return ts

    # create two waypoints with offset from gate center position, with the same yaw rotation as gate
    # can be used to influence trajectory generation to make drone go through the gates in a more straight line
    def create2WaypointsOffset(self, position, yaw, offset):
        # offset vector
        ov1 = np.array([0,offset,0])
        ov2 = np.array([0,offset,0])
        # rotation matrix z-axis
        r1 = np.array([[cos(yaw), sin(yaw), 0], [-1*sin(yaw), cos(yaw), 0], [0, 0, 1]])
        yaw += pi
        r2 = np.array([[cos(yaw), sin(yaw), 0], [-1*sin(yaw), cos(yaw), 0], [0, 0, 1]])
        rot1 = r1.dot(np.transpose(ov1))
        rot2 = r2.dot(np.transpose(ov2))
        pos1 = deepcopy(position) + rot1
        pos2 = deepcopy(position) + rot2
        return (pos1, pos2)

    # gets current position of gates and generates a trajectory through those points
    # if traj is true:
    #   returns maveric.Waypoint objects, trajectory as list of parameters for polynomials (see convertTrajectoryToWaypoints())
    # else: 
    #   returns waypoints as list of [x, y, z, yaw], yaw in degrees
    def generateTrajectoryFromCurrentGatePositions(self, timestep=1, traj=True):
        
        waypoints = []

        # uav current position is first waypoint
        uavpos = self.getPositionUAV()
        orientation = self.getPositionAirsimUAV().orientation
        # convert to xyz and yaw
        _, _, yaw = airsim.to_eularian_angles(orientation)
        uavwp = [uavpos[0], uavpos[1], uavpos[2], degrees(-yaw)]
        # add waypoint
        waypoints.append(uavwp)

        # get current gate positions
        for i in range(1,5):
            # get gate position
            gp = self.getPositionGate(i)
            # self.printPose(gp)
            orientation = self.getPositionAirsimGate(i).orientation
            # convert to xyz and yaw
            _, _, yaw = airsim.to_eularian_angles(orientation)

            wp1, wp2 = self.create2WaypointsOffset(gp[:3], -yaw, 1)
            wp1 = [wp1[0], wp1[1], wp1[2], degrees(-yaw)]
            wpg = [gp[0], gp[1], gp[2], degrees(-yaw)]
            wp2 = [wp2[0], wp2[1], wp2[2], degrees(-yaw)]
            # add waypoint
            # waypoints.append(wp2)
            waypoints.append(wpg)
            # waypoints.append(wp1)

        # add uavwp again - starting point as endpoint
        waypoints.append(uavwp)


        if traj:
            # call maveric to get trajectory
            return maveric.planner(waypoints)  # timestep=timestep
        else:
            # return list of waypoints
            return waypoints 

    # this is a sampling function which
    # converts waypoints, trajectory from generateTrajectoryFromCurrentGatePositions() to airsim.Vector3r list and [x,y,z,yaw,timestamp]
    # config.waypoints_per_segment waypoints will be generated for each segment of trajectory, where a segment is the polynomial between two waypoints (gates)
    def convertTrajectoryToWaypoints(self, waypoints, trajectory, evaltime=10):
        out = []
        outComplete = []

        # for each segment...
        for i in range(len(trajectory)):

            t = np.linspace(waypoints[i].time, waypoints[i + 1].time, self.config.waypoints_per_segment)
            x_path = (trajectory[i][0] * t ** 4 + trajectory[i][1] * t ** 3 + trajectory[i][2] * t ** 2 + trajectory[i][3] * t + trajectory[i][4])
            y_path = (trajectory[i][5] * t ** 4 + trajectory[i][6] * t ** 3 + trajectory[i][7] * t ** 2 + trajectory[i][8] * t + trajectory[i][9])
            z_path = (trajectory[i][10] * t ** 4 + trajectory[i][11] * t ** 3 + trajectory[i][12] * t ** 2 + trajectory[i][13] * t + trajectory[i][14])
            yaw_path = (trajectory[i][15] * t ** 2 + trajectory[i][16] * t + trajectory[i][17])

            # ... calculate position of each waypoint
            for j in range(len(t)):
                out.append(airsim.Vector3r(x_path[j], y_path[j], z_path[j]))
                yaw = self.vectorToYaw(np.array([x_path[j], y_path[j], z_path[j]]) - np.array([waypoints[i+1].x, waypoints[i+1].y, waypoints[i+1].z]))
                outComplete.append([x_path[j], y_path[j], z_path[j], yaw, t[j]]) # yaw_path[j]

        return out, outComplete

    def vectorToYaw(self, vec):
        return degrees(atan2(vec[1], vec[0])) - 180


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
            # print(f"created folder '{folder}'")
            os.mkdir(folder)

    # reset simulation environment
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)




