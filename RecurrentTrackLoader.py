#!/usr/bin/env python

from cProfile import label
from random import sample
import sys

sys.path.append('../')

import json
import math

import cv2
import torchvision.transforms as transforms

import glob
from pathlib import Path

from itertools import chain

import torch
from torch.utils.data import Dataset
import torch.nn.functional as fn

import util
import numpy as np
import pfm
import orb
import config
'''
loads data for single race track (usually called trackXX with XX in [0:100]
returns generator object
item contains timestamp, waypoints, pose, imu:list, path of left image : string, velocity
'''


class RacetrackLoader:

    def __init__(self, dataset_basepath: str, dataset_basename: str, raceTrackName: str,
                 localTrajectoryLength: int = 3, skipLastXImages=0, skipFirstXImages=0):
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

        self.skipLastXImages = -1 if skipLastXImages == 0 else skipLastXImages * -1
        self.skipFirstXImages = 0 if skipFirstXImages < 1 else skipFirstXImages

    def __iter__(self):

        for i, frame in enumerate(self.data[self.skipFirstXImages:self.skipLastXImages]):
            # load left image
            lipath = self.DATASET_PATH_LEFT + "/" + frame['image_name'] + ".png"
            dpath = self.DATASET_PATH_DEPTH + "/" + frame['image_name'] + ".pfm"

            # load other values
            ts = frame['time_stamp']
            waypoints = self.config.waypoints[
                        frame['waypoint_index']: frame['waypoint_index'] + self.localTrajectoryLength]
            pose = frame['pose']
            imu = frame['imu']
            vel = frame['body_velocity_yaw_pid']
            Wvel = frame['world_velocity_yaw_pid']

            yield ts, waypoints, pose, imu, lipath, dpath, vel, Wvel

    def loadConfig(self, configFile):
        class ConfigLoader:
            def __init__(self, **data):
                self.__dict__.update(data)

        data = json.load(configFile)
        self.config = ConfigLoader(**data)

    def __del__(self):
        self.configFile.close()

    # export drone and ground truth trajectory in TUM file/data format
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


'''
loads all race tracks in a dataset folder
returns torch.utils.data.Dataset
item returns left image and velocity vector in local body frame FLU as label
'''


class RaceTracksDataset(Dataset):
    def __init__(self, dataset_basepath: str, dataset_basename: str, device='cpu',
                 yawMaxCommand=10, skipTracks=0, maxTracksLoaded=-1, imageScale=100, grayScale=True,
                 # imageScale in percent of original image size
                 imageTransforms=None, loadRGB=True, loadDepth=False, loadOrb=False, orb_features=1000,
                                                                              *args, **kwargs
                 ):

        # create image transform to transform image to tensor
        self.imageTransform = imageTransforms

        dataset_path = dataset_basepath + "/" + dataset_basename
        # load all tracks in directory
        self.rtLoaders = []
        loadedTracks = 0
        if maxTracksLoaded == -1:
            maxTracksLoaded = math.inf
        for path in glob.glob(f"{dataset_path}/*/"):
            if loadedTracks >= maxTracksLoaded + skipTracks:
                break
            loadedTracks += 1
            if loadedTracks <= skipTracks:
                continue
            trackname = Path(path).parts[-1]
            self.rtLoaders.append(RacetrackLoader(dataset_basepath, dataset_basename, trackname, *args, **kwargs))

        self.data = list(chain(*self.rtLoaders))

        # filter out data points with too high velocity
        # self.data = list(filter(lambda x: util.magnitude(np.array(x[5][:3])) > max_velocity, self.data))

        self.device = device
        self.yawMaxCommand = yawMaxCommand
        self.imageScale = imageScale
        self.grayScale = grayScale
        self.first = True
        self.loadRGB = loadRGB
        self.loadD = loadDepth
        self.loadOrb = loadOrb
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.orb_features_cache = [None for _ in range(len(self.data))]

    def __getitem__(self, index):
        _, _, _, _, lipath, dpath, velocity, Wvelocity = self.data[index]
        label = torch.tensor(velocity, dtype=torch.float32)
        sample = self.loadSample(lipath, index, dpath)
        # move to device
        label = label.to(self.device)
        sample = sample.to(self.device)
        return sample, label

    def __len__(self):
        return len(self.data)

    def loadSample(self, imagePath, index, depthpath=None):

        sample = None

        if self.loadRGB or self.loadOrb:
            image = cv2.imread(imagePath)

        # load image
        if self.loadRGB:
            sample = self.loadImage(image)

        if self.loadD:
            if depthpath is None:
                raise Exception("No depth path given")
            depthimage = self.loadDepth(depthpath)

            if sample is not None:
                sample = torch.cat((sample, depthimage), 0)
            else:
                sample = depthimage

        # apply transforms before adding orb mask
        if self.imageTransform:
            sample = self.imageTransform(sample)

        if self.loadOrb:
            orbmask = self.calculateOrbMask(image, index)
            if sample is not None:
                sample = torch.cat((sample, orbmask), 0)
            else:
                sample = orbmask

        return sample

    def calculateOrbMask(self, image, index):

        if self.orb_features_cache[index] is None:
            kp, des, _, _, _ = orb.get_orb(image)
            self.orb_features_cache[index] = (kp, des)
        else:
            kp, des = self.orb_features_cache[index]
        # convert to torch tensor
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)
        orbmask = torch.zeros_like(image[0])
        for el in kp:
            x, y = el.pt
            orbmask[int(y), int(x)] = 1
        orbmask = orbmask.unsqueeze(0)
        return orbmask


    def loadDepth(self, depthpath):
        depthimage = pfm.read_pfm(depthpath)
        depthimage = depthimage[0]
        depthimage = transforms.Compose([
            transforms.ToTensor(),
        ])(depthimage)
        return depthimage

    def loadImage(self, image):

        if self.grayScale:
            # not tested after recent changes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale down
        if not self.imageScale == 100:
            # not tested after recent changes
            scale_percent = self.imageScale  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # convert to torch tensor
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        return image


    '''
    calculate velocity vector
    x,y,z velocity = normalized mean of next n waypoints, moved to local frame
    where n = localTrajectoryLength
    forward left up (x, y, z)
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
        yaw *= 1. / float(self.yawMaxCommand)
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


'''
Recurrent Track dataset
    Arg: Datasetpath
    Step: iter through dataset

    Return:  ts, waypoints, pose, imu, lipath, dpath, vel, Wvel

'''

class RecurrentTrackLoader:

    def __init__(self, dataset_basepath: str, dataset_basename: str, raceTrackName: str,
                 localTrajectoryLength: int = 3, skipLastXImages=0, skipFirstXImages=0, sequencelength=5):
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

        self.skipLastXImages = -1 if skipLastXImages == 0 else skipLastXImages * -1
        self.skipFirstXImages = 0 if skipFirstXImages < 1 else skipFirstXImages
        self.sequence_length = sequencelength

    def __iter__(self):
        for i in range(self.skipFirstXImages,len(self.data[:-(self.sequence_length + self.skipFirstXImages)]),5):
        # for i, frame in enumerate(self.data[self.skipFirstXImages:self.skipLastXImages]):
            # load left image
            
            frame_i = self.data[i:i+self.sequence_length]
            lipath = []
            dpath = []
            ts = []
            waypoints = []
            pose = []
            imu = []
            vel = []
            Wvel = []
            # seq_len = self.sequence_length
            for j, frame_j in enumerate(frame_i):
                try:
                    frame = frame_i[j]
                    # print(i + j)
                except:
                    pass
                lipath.append(self.DATASET_PATH_LEFT + "/" + frame['image_name'] + ".png")
                dpath.append(self.DATASET_PATH_DEPTH + "/" + frame['image_name'] + ".pfm")

                # load other values
                ts.append(frame['time_stamp'])
                waypoints.append(self.config.waypoints[
                            frame['waypoint_index']: frame['waypoint_index'] + self.localTrajectoryLength])
                pose.append(frame['pose'])
                imu.append(frame['imu'])
                vel.append(frame['body_velocity_yaw_pid'])
                Wvel.append(frame['world_velocity_yaw_pid'])

                yield ts, waypoints, pose, imu, lipath, dpath, vel, Wvel

    def loadConfig(self, configFile):
        class ConfigLoader:
            def __init__(self, **data):
                self.__dict__.update(data)

        data = json.load(configFile)
        self.config = ConfigLoader(**data)

    def __del__(self):
        self.configFile.close()

    # export drone and ground truth trajectory in TUM file/data format


'''
loads all race tracks in a dataset folder
returns torch.utils.data.Dataset
item returns left image and velocity vector in local body frame FLU as label
'''
class RecurrentTracksDataset(Dataset):
    def __init__(self, dataset_basepath: str, dataset_basename: str, device='cpu',
                 yawMaxCommand=10, skipTracks=0, maxTracksLoaded=-1, imageScale=100, grayScale=True,
                 # imageScale in percent of original image size
                 imageTransforms=None, loadRGB=True, loadDepth=False, loadOrb=False, orb_features=1000,
                                                                              *args, **kwargs
                 ):

        # create image transform to transform image to tensor
        self.imageTransform = imageTransforms

        dataset_path = dataset_basepath + "/" + dataset_basename
        # load all tracks in directory
        self.rtLoaders = []
        loadedTracks = 0
        if maxTracksLoaded == -1:
            maxTracksLoaded = math.inf
        for path in glob.glob(f"{dataset_path}/*/"):
            if loadedTracks >= maxTracksLoaded + skipTracks:
                break
            loadedTracks += 1
            if loadedTracks <= skipTracks:
                continue
            trackname = Path(path).parts[-1]
            self.rtLoaders.append(RecurrentTrackLoader(dataset_basepath, dataset_basename, trackname, *args, **kwargs))

        self.data = list(chain(*self.rtLoaders))

        # filter out data points with too high velocity
        # self.data = list(filter(lambda x: util.magnitude(np.array(x[5][:3])) > max_velocity, self.data))

        self.device = device
        self.yawMaxCommand = yawMaxCommand
        self.imageScale = imageScale
        self.grayScale = grayScale
        self.first = True
        self.loadRGB = loadRGB
        self.loadD = loadDepth
        self.loadOrb = loadOrb
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.orb_features_cache = [None for _ in range(len(self.data))]

    def __getitem__(self, index):
        label = []
        sample = []
        # lipaths = self.data[index][-2]
        # (ts, waypoints, pose, imu, lipath, vel)
        
        _, _, _, _, lipaths, dpaths, velocity, Wvelocity = self.data[index]
        for i, lipath in enumerate(lipaths):
            labels = torch.tensor(velocity[i], dtype=torch.float32)
            samples = self.loadSample(lipath, i, dpaths[i])
            # move to device
            sample.append(samples.unsqueeze(0))
            label.append(labels.unsqueeze(0))


        sample = torch.cat(sample, axis=0)
        label = torch.cat(label, axis=0)
        return sample, label

    def __len__(self):
        return len(self.data)

    def loadSample(self, imagePath, index, depthpath=None):

        sample = None

        if self.loadRGB or self.loadOrb:
            image = cv2.imread(imagePath)

        # load image
        if self.loadRGB:
            sample = self.loadImage(image)

        if self.loadD:
            if depthpath is None:
                raise Exception("No depth path given")
            depthimage = self.loadDepth(depthpath)

            if sample is not None:
                sample = torch.cat((sample, depthimage), 0)
            else:
                sample = depthimage

        # apply transforms before adding orb mask
        if self.imageTransform:
            sample = self.imageTransform(sample)

        if self.loadOrb:
            orbmask = self.calculateOrbMask(image, index)
            if sample is not None:
                sample = torch.cat((sample, orbmask), 0)
            else:
                sample = orbmask

        return sample

    def calculateOrbMask(self, image, index):

        if self.orb_features_cache[index] is None:
            kp, des, _, _, _ = orb.get_orb(image)
            self.orb_features_cache[index] = (kp, des)
        else:
            kp, des = self.orb_features_cache[index]
        # convert to torch tensor
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)
        orbmask = torch.zeros_like(image[0])
        for el in kp:
            x, y = el.pt
            orbmask[int(y), int(x)] = 1
        orbmask = orbmask.unsqueeze(0)
        return orbmask


    def loadDepth(self, depthpath):
        depthimage = pfm.read_pfm(depthpath)
        depthimage = depthimage[0]
        depthimage = transforms.Compose([
            transforms.ToTensor(),
        ])(depthimage)
        return depthimage

    def loadImage(self, image):

        if self.grayScale:
            # not tested after recent changes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale down
        if not self.imageScale == 100:
            # not tested after recent changes
            scale_percent = self.imageScale  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # convert to torch tensor
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        return image


    '''
    calculate velocity vector
    x,y,z velocity = normalized mean of next n waypoints, moved to local frame
    where n = localTrajectoryLength
    forward left up (x, y, z)
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
        yaw *= 1. / float(self.yawMaxCommand)
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


'''
Recurrent Track dataset
    Arg: Datasetpath
    Step: iter through dataset

    Return:  ts, waypoints, pose, imu, lipath, dpath, vel, Wvel

'''

if __name__ == "__main__":
    dataset_basepath ='/media/data2/teamICRA/X4Gates_Circles_rl18tracks'
    dataset_basename = 'X1Gate_dagger_dung'
    epochs = 100
    batch_size = 32
    # train_tracks = [0]
    train_tracks = 180
    device = 'cuda'
    skipFirstXImages = 3
    skipLastXImages = 0
    tf = config.tf
    maxTracks = config.num_train_tracks
    # val_tracks = [8,3]
    input_channels = config.input_channels
    dataset = RecurrentTracksDataset(
                    dataset_basepath,
                    dataset_basename,
                    device=device,
                    maxTracksLoaded=maxTracks,
                    imageScale=100,
                    skipTracks=0,
                    grayScale=False,
                    imageTransforms=tf,
                    skipLastXImages=skipLastXImages,
                    skipFirstXImages=skipFirstXImages,
                    loadRGB=input_channels['rgb'],
                    loadDepth=input_channels['depth'],
                    loadOrb=input_channels['orb']
                    )
    datasetloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    # log_df = pd.DataFrame({
    #     'x': [],
    #     'y': [],
    #     'z': [],
    #     'yaw': []
    # })
    log_df = []
    for spls, lbs in datasetloader:
        for i , spl in enumerate(spls):
            print(spl)