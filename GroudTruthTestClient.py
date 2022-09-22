#!/usr/bin/env python

'''

simulation client for AirSimInterface.py
this runs the main loop and holds the settings for the simulation. 


'''

import sys
from urllib import response

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from AirSimInterface import AirSimInterface
from SimClient import SimClient

import airsim
import numpy as np
import pprint
import curses
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn

import os
import time
from math import *
import time

import cv2
from copy import deepcopy

import imitation.ResNet8 as resnet8

# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
# from VelocityPID import VelocityPID
from UnityPID import VelocityPID

from util import *

from RaceTrackLoader import RaceTracksDataset, RacetrackLoader

torch.set_grad_enabled(False)


class GroundTruthTestClient(SimClient):

    def __init__(self, modelPath, raceTrackName="track0", device='cpu'):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False)

        # do custom setup here

        self.gateConfigurations = []
        self.currentGateConfiguration = 0


        device = 'cpu'
        batch_size = 16

        skipLastXImages = 0

        # dataset_basepath = "/media/micha/eSSD/datasets"
        dataset_basepath = self.config.dataset_basepath
        # dataset_basepath = "/data/datasets"
        # dataset_basename = "X4Gates_Circle_right_"
        dataset_basename = self.DATASET_NAME
        # dataset_basename = "X4Gates_Circle_2"


        # relead config file from dataset
        # configuration file
        self.configFile = open(f'{dataset_basepath}/{dataset_basename}/track0/config.json', "r")

        self.config = {}
        self.loadConfig(self.configFile)

        self.loadGatePositions(self.config.gates['poses'])

        print("loading dataset...")

        # ds = {
        #     'train':
        #         torch.utils.data.DataLoader(
        #             RaceTracksDataset(
        #                 dataset_basepath,
        #                 dataset_basename,
        #                 device=device,
        #                 maxTracksLoaded=1,
        #                 imageScale=100,
        #                 skipTracks=0,
        #                 grayScale=False,
        #                 skipLastXImages=skipLastXImages,
        #                 imageTransforms=transforms.Compose([transforms.Resize((200, 300))])
        #             ),
        #             batch_size=1,
        #             shuffle=False
        #         ),
        #     'val':
        #         torch.utils.data.DataLoader(
        #             RaceTracksDataset(
        #                 dataset_basepath,
        #                 dataset_basename,
        #                 device=device,
        #                 maxTracksLoaded=1,
        #                 imageScale=100,
        #                 skipTracks=0,
        #                 grayScale=False,
        #                 skipLastXImages=skipLastXImages,
        #                 imageTransforms=transforms.Compose([transforms.Resize((200, 300))])
        #             ),
        #             batch_size=1,
        #             shuffle=True
        #         )
        # }
        # ds = RaceTracksDataset(
        #                 dataset_basepath,
        #                 dataset_basename,
        #                 device=device,
        #                 maxTracksLoaded=1,
        #                 imageScale=100,
        #                 skipTracks=0,
        #                 grayScale=False,
        #                 skipLastXImages=skipLastXImages,
        #                 imageTransforms=transforms.Compose([transforms.Resize((200, 300))])
        #             )

        self.dataset = iter(RacetrackLoader(dataset_basepath, dataset_basename, "track0", 1,
                                                  skipLastXImages=skipLastXImages))



    def runWorld(self):

        self.client.simPause(False)

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        time.sleep(3)

        lastImage = time.time()

        timePerImage = 1. / float(self.config.framerate)

        cimageindex = 0

        while mission:

            # get current time and time delta
            tn = time.time()

            nextImage = tn - lastImage > timePerImage

            if nextImage:
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)

                try:
                    _, _, _, _, lipath, velocity, Wvelocity = next(self.dataset)

                except StopIteration:
                    mission = False
                    break

                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                # if self.config.debug:
                #     self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                # else:
                #     print(f"pausedelta: {pausedelta}")
                tn += pausedelta
                lastImage = tn

                # send control command to airsim
                cstate = self.getState()

                # # rotate velocity command such that it is in world coordinates
                # Wvel = vector_body_to_world(pred[:3], [0, 0, 0], cstate[3])
                Wvel = Wvelocity[:3]

                # # add pid output for yaw to current yaw position
                # Wyaw = degrees(cstate[3]) - degrees(pred[3])
                Wyaw = degrees(Wvelocity[3])

                # visualizes prediction 
                self.client.simPlotPoints([self.getPositionAirsimUAV().position], color_rgba=[1.0, 0.0, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)
                Wposvel = cstate[:3] + Wvel
                self.client.simPlotPoints([airsim.Vector3r(Wposvel[0], Wposvel[1], Wposvel[2])], color_rgba=[.8, 0.5, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)

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
                self.client.moveByVelocityAsync(float(Wvel[0]), float(Wvel[1]), float(Wvel[2]),
                                                duration=float(timePerImage), yaw_mode=airsim.YawMode(False, Wyaw))



    def runBody(self):

        self.client.simPause(False)

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        time.sleep(3)

        lastImage = time.time()

        timePerImage = 1. / float(self.config.framerate)

        cimageindex = 0

        while mission:

            # get current time and time delta
            tn = time.time()

            nextImage = tn - lastImage > timePerImage

            if nextImage:
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)

                try:
                    _, _, _, _, lipath, velocity, Wvelocity = next(self.dataset)

                except StopIteration:
                    mission = False
                    break

                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                # if self.config.debug:
                #     self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                # else:
                #     print(f"pausedelta: {pausedelta}")
                tn += pausedelta
                lastImage = tn

                # send control command to airsim
                cstate = self.getState()

                # # rotate velocity command such that it is in world coordinates
                Wvel = vector_body_to_world(velocity[:3], [0, 0, 0], cstate[3]) * 2  # velocity limit in sim client = 2
                # Wvel = Wvelocity[:3]

                # # add pid output for yaw to current yaw position
                Wyaw = degrees(cstate[3]) + degrees(velocity[3])
                # Wyaw = degrees(Wvelocity[3])

                # visualizes prediction 
                self.client.simPlotPoints([self.getPositionAirsimUAV().position], color_rgba=[1.0, 0.0, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)
                Wposvel = cstate[:3] + Wvel
                self.client.simPlotPoints([airsim.Vector3r(Wposvel[0], Wposvel[1], Wposvel[2])], color_rgba=[.8, 0.5, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)

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
                self.client.moveByVelocityAsync(float(Wvel[0]), float(Wvel[1]), float(Wvel[2]),
                                                duration=float(timePerImage), yaw_mode=airsim.YawMode(False, Wyaw))




    def loadWithAirsim(self):
        # get images from AirSim API
        res = self.client.simGetImages(
            [
                airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                # airsim.ImageRequest("front_right", airsim.ImageType.Scene),
                # airsim.ImageRequest("depth_cam", airsim.ImageType.DepthPlanar, True)
            ]
        )
        left = res[0]

        img1d = np.fromstring(left.image_data_uint8, dtype=np.uint8)
        image = img1d.reshape(left.height, left.width, 3)
        # image = np.flipud(image) - np.zeros_like(image)  # pytorch conversion from numpy does not support negative stride

        # preprocess image
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        # image = dn.preprocess(image)
        return image


if __name__ == "__main__":
    import contextlib

    print("test with world coordinate velocity")

    with contextlib.closing(GroundTruthTestClient(
            "",
            device="cuda")) as nc:
        nc.runWorld()

    print("test with body coordinate velocity")

    with contextlib.closing(GroundTruthTestClient(
            "",
            device="cuda")) as nc:
        nc.runBody()
