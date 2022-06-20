#!/usr/bin/env python

import json
import cv2
import torchvision.transforms as transforms

import glob
import os
from pathlib import Path

from itertools import chain

import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as fn

import util

import numpy as np

'''

'''


class RacetrackLoader:

    def __init__(self, dataset_basepath: str, dataset_basename: str, raceTrackName: str,
                 localTrajectoryLength: int = 3):
        '''
        INIT VALUES
        '''

        # append run number to base name
        self.DATASET_NAME = dataset_basename
        self.DATASET_PATH_SUPER = dataset_basepath + "/" + self.DATASET_NAME
        self.DATASET_PATH = self.DATASET_PATH_SUPER + "/" + raceTrackName
        self.DATASET_PATH_FILE = self.DATASET_PATH + "/data.json"
        self.DATASET_PATH_LEFT = self.DATASET_PATH + "/image_left"
        self.DATASET_PATH_RIGHT = self.DATASET_PATH + "/image_right"
        self.DATASET_PATH_DEPTH = self.DATASET_PATH + "/image_depth"

        # configuration file
        self.configFile = open(self.DATASET_PATH + "/config.json", "r")

        self.config = {}
        self.loadConfig(self.configFile)

        # load list of items in data file
        with open(self.DATASET_PATH_FILE, "r") as dataFile:
            data = json.load(dataFile)
            self.data = data['data']

        self.localTrajectoryLength = localTrajectoryLength

        self.raceTrackName = raceTrackName

    def __iter__(self):
        for frame in self.data:
            # load left image
            lipath = self.DATASET_PATH_LEFT + "/" + frame['image_name'] + ".png"
            # img = self.loadImage(lipath)

            # load other values
            ts = frame['time_stamp']
            waypoints = self.config.waypoints[
                        frame['waypoint_index']: frame['waypoint_index'] + self.localTrajectoryLength]
            pose = frame['pose']
            imu = frame['imu']

            yield ts, waypoints, pose, imu, lipath

    def loadConfig(self, configFile):
        class ConfigLoader:
            def __init__(self, **data):
                self.__dict__.update(data)

        data = json.load(configFile)
        self.config = ConfigLoader(**data)

    def __del__(self):
        self.configFile.close()

    def exportTrajectoriesAsTum(self):
        # output file names
        fp1, fp2 = \
            (
                self.DATASET_PATH + f"/groundtruth_trajectory_{self.raceTrackName}.tum",
                self.DATASET_PATH + f"/actual_trajectory_{self.raceTrackName}.tum"
            )
        with open(fp1, 'w') as gtf, open(fp2, 'w') as trf:
            for d in self.data:
                ts = float(d['time_stamp'])
                # convert to tum float timestamp, seconds after epoch
                ts *= 1e-9

                # get waypoint for current pose
                wp = self.config.waypoints[d['waypoint_index']]

                # convert yaw to quaternion
                orientation = list(util.to_quaternion(0, 0, wp[3]))
                wp = wp[:3]
                orientation = orientation[1:] + orientation[:1]
                wp += orientation

                # get current pose
                pose = d['pose']

                # write lines to file
                # ts, x, y, z, qx, qy, qz, qw
                gtf.write(" ".join([str(el) for el in [ts, *pose]]))
                gtf.write("\n")
                trf.writelines(" ".join([str(el) for el in [ts, *wp]]))
                trf.write("\n")


class RaceTracksDataset(Dataset):
    def __init__(self, dataset_basepath: str, dataset_basename: str, localTrajectoryLength: int = 3, device='cpu', yawMaxCommand=10):

        # create image transform to transform image to tensor
        self.imageTransform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset_path = dataset_basepath + "/" + dataset_basename
        # load all tracks in directory
        self.rtLoaders = []
        for path in glob.glob(f"{dataset_path}/*/"):
            trackname = Path(path).parts[-1]
            self.rtLoaders.append(RacetrackLoader(dataset_basepath, dataset_basename, trackname, localTrajectoryLength))

        self.data = list(chain(*self.rtLoaders))

        self.device = device

        self.yawMaxCommand = yawMaxCommand

    def __getitem__(self, index):
        ts, waypoints, pose, imu, lipath = self.data[index]
        if index > 0:
            _, _, prevPose, _, _ = self.data[index - 1]
        else:
            prevPose = pose
        label = self.waypointToVelocityVector(waypoints, pose, prevPose)
        # TODO: normalize?
        sample = self.loadImage(lipath)
        # move to device
        label = label.to(self.device)
        sample = sample.to(self.device)
        return sample, label

    def __len__(self):
        return len(self.data)

    def loadImage(self, path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # scale down
        scale_percent = 30  # percent of original size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

        tensor = self.imageTransform(resized)
        return tensor


    '''
    calculate velocity vector
    x,y,z velocity = normalized mean of next n waypoints, moved to local frame
    where n = localTrajectoryLength
    '''
    def waypointToVelocityVector(self, waypoints, pose, prevPose):
        ret = torch.tensor(waypoints, dtype=torch.float32)
        # remove timestamp
        ret = ret[:, :4]
        # move waypoints into local frame
        posey = torch.tensor([pose[0], pose[1], pose[2], util.to_eularian_angles(*pose[3:7])[2]])
        prevPosey = torch.tensor([prevPose[0], prevPose[1], prevPose[2], util.to_eularian_angles(*prevPose[3:7])[2]])
        ret = ret - posey
        # calculate unit vector from local waypoints
        ret = torch.mean(ret, dim=0)
        # calculate actual yaw change since last frame
        yaw = posey[3] - prevPosey[3]
        # if yaw request is too high then there was probably a jump to the next race track in the dataset
        if -self.yawMaxCommand > yaw or yaw > self.yawMaxCommand:
            yaw = 0
        # normalize yaw
        yaw *= 1./float(self.yawMaxCommand)
        ret = ret[:3]
        ret = fn.normalize(ret, dim=0)
        return torch.tensor([*ret, yaw])

    def exportRaceTracksAsTum(self):
        for el in self.rtLoaders:
            el.exportTrajectoriesAsTum()

    def getWaypoints(self):
        ret = []
        for el in self.rtLoaders:
            ret += el.config.waypoints
        return ret

    def getPoses(self, skip=0):
        poses = []
        velocities = []
        for index in range(len(self.data)):
            ts, waypoints, pose, imu, lipath = self.data[index]
            if index > 0:
                _, _, prevPose, _, _ = self.data[index - 1]
            else:
                prevPose = pose
            vector = self.waypointToVelocityVector(waypoints, pose, prevPose)
            poses.append(pose)
            velocities.append(vector)

        if skip <= 0:
            return poses, velocities
        else:
            return poses[::skip], velocities[::skip]
