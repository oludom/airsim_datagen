#!/usr/bin/env python

import sys
from turtle import color
import numpy as np
from tqdm import tqdm

import airsim
import numpy

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

from util import vector_body_to_world
import pandas as pd
import matplotlib.pyplot as plt
'''
loads data for single race track (usually called trackXX with XX in [0:100]
returns generator object
item contains timestamp, waypoints, pose, imu:list, path of left image : string, velocity
'''


class RacetrackLoader:

    def __init__(self, dataset_basepath: str, dataset_basename: str, raceTrackName: str,
                 localTrajectoryLength: int = 3, skipFirstXImages = 0,skipLastXImages=0):
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
        self.skipFirstXImages = 0 if skipFirstXImages == 0 else skipFirstXImages * 1
        self.skipLastXImages = -1 if skipLastXImages == 0 else skipLastXImages * -1

    def __iter__(self):

        for i, frame in enumerate(self.data[self.skipFirstXImages : self.skipLastXImages]):
            # load left image
            lipath = self.DATASET_PATH_LEFT + "/" + frame['image_name'] + ".png"

            # load other values
            ts = frame['time_stamp']
            waypoints = self.config.waypoints[
                        frame['waypoint_index']: frame['waypoint_index'] + self.localTrajectoryLength]
            pose = frame['pose']
            imu = frame['imu']
            vel = frame['body_velocity_yaw_pid']

            yield ts, waypoints, pose, imu, lipath, vel

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
    def __init__(self, dataset_basepath: str, dataset_basename: str, localTrajectoryLength: int = 3, device='cpu',
                 yawMaxCommand=10, skipTracks=0, maxTracksLoaded=-1, imageScale=100, grayScale=True,
                 # imageScale in percent of original image size
                 imageTransforms=None, skipFirstXImages =0, skipLastXImages=0, train=True, tracknames=[0]
                 ):
        """
        args:
            + maxTracksLoaded: Number of tracks folder to be loaded into the dataset
            + skipLastXImages: Number of images to skip (from the end of each folder)
            + skipTracks: Number of tracks (folder) to skip from the data directory
        """

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

            trackname = Path(path).parts[-1]
            loadedTracks +=1
            if loadedTracks <= skipTracks:
                continue
            
            # if int(trackname[-1]) in tracknames:
            # a RacetrackLoader is used for only a track folder
            self.rtLoaders.append(RacetrackLoader(dataset_basepath, dataset_basename, trackname, localTrajectoryLength, skipFirstXImages = skipFirstXImages,
                                                skipLastXImages=skipLastXImages))
            training_mode = "training" if train else "validation"
            print(f"Loaded {trackname} for {training_mode}")
            



        # Chain all tracks samples into a single list of multiple tuples: (ts, waypoints, pose, imu, lipath, vel)
        self.data = list(chain(*self.rtLoaders))

        self.device = device
        self.yawMaxCommand = yawMaxCommand
        self.imageScale = imageScale
        self.grayScale = grayScale
        self.first = True
        self.max_v = None
        self.find_max_velocities()

    def find_max_velocities(self):
        print("finding maximum absolute values of x,y,z,yaw velocities...")
        velocities = np.zeros([len(self.data), 4]) 
        for i, sample in enumerate(tqdm(self.data)):
            _, _, _, _, lipath, velocity = sample
            velocities[i, :] = velocity
        self.max_v = torch.tensor(np.max(abs(velocities), axis=0), dtype=torch.float32)
        print(f"Maximum velociies: {self.max_v}")
        print(self.max_v.shape)
        print("Done")
    
    @staticmethod
    def to_eularian_angles(q):
        # z = q.z_val
        # y = q.y_val
        # x = q.x_val
        # w = q.w_val
        # print(q)
        w, x, y, z = q[0], q[1], q[2], q[3] 
        
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)


    def __getitem__(self, index):
        
        # (ts, waypoints, pose, imu, lipath, vel)
        _, _, pose, _, lipath, velocity = self.data[index]
        #convert body velocity and yaw difference to world frame
        rots = self.to_eularian_angles(pose[3:]) # pitch roll yaw (in radians)
        bvel, byaw = velocity[:3], velocity[3] # byaw is in radians
        wvel = vector_body_to_world(bvel, [0, 0, 0], rots[2])
        wyaw = rots[2] + byaw  # wyaw is in radians
        w_velocity = [*wvel, wyaw]
        # print('adasdasdas', w_velocity)
        # Convert to tensor
        label = torch.tensor(w_velocity, dtype=torch.float32)
        # label = label / self.max_v
        sample = self.loadImage(lipath) 
        # print(label)
        return sample, label

    def __len__(self):
        return len(self.data)

    def loadImage(self, path):
        image = cv2.imread(path)

        if self.grayScale:
            # not tested after recent changes
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # scale down
        if not self.imageScale == 100:
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

        # apply transforms
        if self.imageTransform:
            image = self.imageTransform(image)

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

class RecurrentRaceTracksLoader(RacetrackLoader):
    def __init__(self, dataset_basepath: str, dataset_basename: str, raceTrackName: str, localTrajectoryLength: int = 3, skipFirstXImages = 0, skipLastXImages=0, sequence_length=5):
        super().__init__(dataset_basepath, dataset_basename, raceTrackName, localTrajectoryLength, skipFirstXImages ,skipLastXImages)
        self.sequence_length = sequence_length

    def __iter__(self):
        for i, frame_i in enumerate(self.data[self.skipFirstXImages:self.skipLastXImages:self.sequence_length]):
            
            lipath = []
            ts = []
            waypoints = []
            pose = []
            imu = []
            vel = []
            
            for j in range(self.sequence_length):
                frame = self.data[i + j]
                
                # load left image
                lipath.append(self.DATASET_PATH_LEFT + "/" + frame['image_name'] + ".png")
                # load other values
                ts.append(frame['time_stamp'])
                waypoints.append(self.config.waypoints[
                            frame['waypoint_index']: frame['waypoint_index'] + self.localTrajectoryLength])    
                pose.append(frame['pose'])
                imu.append(frame['imu'])
                vel.append(frame['body_velocity_yaw_pid'])

            yield ts, waypoints, pose, imu, lipath, vel

class RecurrentRaceTrackDataset(RaceTracksDataset):
    def __init__(self, dataset_basepath: str, dataset_basename: str, localTrajectoryLength: int = 3, device='cpu', yawMaxCommand=10, skipTracks=0, maxTracksLoaded=-1, imageScale=100, grayScale=True, imageTransforms=None, skipFirstXImages = 0,skipLastXImages=0, tracknames = [0], train=True):
        # super().__init__(dataset_basepath, dataset_basename, localTrajectoryLength, device, yawMaxCommand, skipTracks, maxTracksLoaded, imageScale, grayScale, imageTransforms, skipLastXImages)
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

            loadedTracks +=1
            
            if loadedTracks <= skipTracks:
                            continue

            
            trackname = Path(path).parts[-1]
            # if int(trackname[-1]) in tracknames:
                # a RacetrackLoader is used for only a track folder
            self.rtLoaders.append(RecurrentRaceTracksLoader(dataset_basepath, dataset_basename, trackname, localTrajectoryLength,skipFirstXImages= skipFirstXImages,
                                                skipLastXImages=skipLastXImages))
            training_mode = "training" if train else "validation"
            print(f"Loaded {trackname} for {training_mode}")
            
            

        # Chain all tracks samples into a single list of multiple tuples: (ts, waypoints, pose, imu, lipath, vel)
        self.data = list(chain(*self.rtLoaders))

        self.device = device
        self.yawMaxCommand = yawMaxCommand
        self.imageScale = imageScale
        self.grayScale = grayScale
        self.first = True


    def __getitem__(self, index):
        label = []
        sample = []
        lipaths = self.data[index][-2]
        # (ts, waypoints, pose, imu, lipath, vel)
        _, _, pose, _, lipath, velocity = self.data[index]
        # label = label / self.max_v
        
        for i, lipath in enumerate(lipaths):
            sample.append(self.loadImage(lipath).unsqueeze(0))
             #convert body velocity and yaw difference to world frame
            rots = self.to_eularian_angles(pose[i][3:]) # pitch roll yaw (in radians)
            bvel, byaw = velocity[i][:3], velocity[i][3] # byaw is in radians
            wvel = vector_body_to_world(bvel, [0, 0, 0], rots[2])
            wyaw = rots[2] + byaw  # wyaw is in radians
            w_velocity = [*wvel, wyaw]
            # Convert to tensor
            label.append(torch.tensor(w_velocity, dtype=torch.float32).unsqueeze(0))
        
        # print(label)
        sample = torch.cat(sample, axis=0)
        label = torch.cat(label, axis=0)  

        ################ Old loader #####################
        # label = torch.tensor(self.data[index][-1], dtype=torch.float32)
        # lipaths = self.data[index][-2]
        # sample = []

        # for i, lipath in enumerate(lipaths):
        #     sample.append(self.loadImage(lipath).unsqueeze(0))
        
        # sample = torch.cat(sample, axis=0)

        return sample, label

if __name__ == "__main__":
    dataset_basepath ='/media/data2/teamICRA/X4Gates_Circles_rl18tracks'
    dataset_basename = 'Onegate-l-3'
    epochs = 100
    batch_size = 32
    train_tracks = [0]
    device = 'cuda'
    skipFirstXImages =0
    skipLastXImages = 0
    # val_tracks = [8,3]
    dataset = RaceTracksDataset(
                        dataset_basepath,
                        dataset_basename,
                        device=device,
                        maxTracksLoaded=len(train_tracks),
                        imageScale=100,
                        skipTracks=0,
                        grayScale=False,
                        skipFirstXImages= skipFirstXImages,
                        skipLastXImages=skipLastXImages,
                        tracknames=train_tracks
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
    for s, l in datasetloader:
        l_np = l.numpy()
        df = pd.DataFrame(l_np)
        log_df.append(df)
        # print(type(log_df))
        # np_arr = numpy.asarray(log_df)
        # log_df = pd.DataFrame(np_arr)
        
        # print(log_df)
    log_df = pd.concat(log_df)
    log_df.to_csv("/media/data2/teamICRA/X4Gates_Circles_rl18tracks/Onegate-l/log.csv", index= None)
    


    # def plot_world(df: pd.DataFrame):
    #     x, y, z, yaw = df['x'], df['y'], df['z'], df['yaw'],  
    #     plt.plot(range(len(x)), x, 'r-')
    #     # plt.plot(range(len(x)), y, 'b-')
    #     # plt.plot(range(len(x)), z, 'g-')
    #     # plt.plot(range(len(x)), , )
    #     plt.show()


    # df = pd.read_csv("/media/data2/teamICRA/X4Gates_Circles_rl18tracks/Onegate-l/log.csv", delimiter= ",")
    # plot_world(df)    
    
    # for epoch in epochs:

        
    
    
    
    # def __init__(self, dataset_basepath: str, dataset_basename: str, localTrajectoryLength: int = 3, device='cpu',
    #              yawMaxCommand=10, skipTracks=0, maxTracksLoaded=-1, imageScale=100, grayScale=True,
    #              # imageScale in percent of original image size
    #              imageTransforms=None, skipLastXImages=0, train=True, tracknames=[0]
    #              ):