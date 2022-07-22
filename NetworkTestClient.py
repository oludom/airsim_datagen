#!/usr/bin/env python

'''

simulation client for AirSimController.py 
this runs the main loop and holds the settings for the simulation. 


'''

import sys
from urllib import response

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from AirSimController import AirSimController
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
        if device=='cuda':
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


        lastWP = time.time()
        lastImage = time.time()
        lastIMU = time.time()
        lastPID = time.time()


        # timePerWP = float(self.config.roundtime) / len(pathComplete)
        timePerImage = 1./float(self.config.framerate)
        timePerIMU = 1./float(self.config.imuRate)
        timePerPID = 1./float(self.config.pidRate)

        cwpindex = 0
        cimageindex = 0

        while mission:
            
            # get current time and time delta
            tn = time.time()

           # nextWP = tn - lastWP > timePerWP
            nextImage = tn - lastImage > timePerImage
            nextIMU = tn - lastIMU > timePerIMU
            nextPID = tn - lastPID > timePerPID

            # if nextIMU:
            #     self.captureIMU()
            #     lastIMU = tn

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
                pred = list(pred[0]*5)

                cimageindex +=1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                if self.config.debug:
                    self.c.addstr(10,0, f"pausedelta: {pausedelta}")
                else:
                    print(f"pausedelta: {pausedelta}")
                lastWP += pausedelta
                lastIMU += pausedelta
                tn += pausedelta
                lastImage = tn

                # send control command to airsim
                cstate = self.getState()

                yaw = cstate[3] + pred[3]
                yaw = float (yaw)

                self.client.moveByVelocityAsync(float(pred[0]), float(pred[1]), float(pred[2]), duration=float(timePerImage), yaw_mode=airsim.YawMode(False, yaw))



            # if self.config.debug:
            #     self.c.addstr(0,0, "following generated path from gates...")
            #     self.c.addstr(2,0, f"frame rate: {hz}")


            if False and nextPID:

                # get current state
                cstate = self.getState()
                # convert yaw to degree
                cstate[3] = degrees(cstate[3])
                # inform pid controller about state
                ctrl.setState(cstate)

                # set goal state of pid controller
                ctrl.setGoal(wp[:4])
                # update pid controller
                ctrl.update(tn - lastPID)
                # get current pid output´
                vel, yaw = ctrl.getVelocityYaw()

                # add pid output for yaw to current yaw position
                yaw = cstate[3] + yaw

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
                self.client.moveByVelocityAsync(float(vel[0]), float(vel[1]), float(vel[2]), duration=float(timePerPID), yaw_mode=airsim.YawMode(False, yaw))

                # save last PID time
                lastPID = tn
        


    
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

    with contextlib.closing(NetworkTestClient("/home/kristoffer/dev/imitation/datagen/eval/runs/ResNet8_bs=32_lt=MSE_lr=0.01_c=run0/best.pth", device="cuda")) as nc:
        nc.loadGatePositions(nc.config.gates['poses'])
        nc.run()