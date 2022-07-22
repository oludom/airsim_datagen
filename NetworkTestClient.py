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


class NetworkTestClient(SimClient):

    def __init__(self, modelPath, raceTrackName="track0", device='cpu'):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False)

        # do custom setup here

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        self.model = resnet8.ResNet8(input_dim=3, output_dim=4)
        if device == 'cuda':
            self.model = nn.DataParallel(self.model)
            cudnn.benchmark = True

        self.model.load_state_dict(torch.load(modelPath))

        self.device = device
        self.dev = torch.device(device)

        self.model.to(self.dev)
        self.model.eval()

    def run(self):

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

                # save images of current frame
                # self.captureAndSaveImages(cwpindex, cimageindex)

                # get images from AirSim API

                image = self.loadWithAirsim()

                images = torch.unsqueeze(image, dim=0)
                images = images.to(self.dev)

                # predict vector with network
                pred = self.model(images)
                pred = list(pred[0] * 5)

                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                if self.config.debug:
                    self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                else:
                    print(f"pausedelta: {pausedelta}")
                tn += pausedelta
                lastImage = tn

                # send control command to airsim
                cstate = self.getState()

                yaw = cstate[3] + pred[3]
                yaw = float(yaw)

                self.client.moveByVelocityAsync(float(pred[0]), float(pred[1]), float(pred[2]),
                                                duration=float(timePerImage), yaw_mode=airsim.YawMode(False, yaw))

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

    with contextlib.closing(NetworkTestClient(
            "/home/kristoffer/dev/imitation/datagen/eval/runs/ResNet8_bs=32_lt=MSE_lr=0.01_c=run0/best.pth",
            device="cuda")) as nc:
        nc.loadGatePositions(nc.config.gates['poses'])
        nc.run()
