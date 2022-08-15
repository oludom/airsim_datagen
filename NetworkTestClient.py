#!/usr/bin/env python

'''

simulation client for AirSimInterface.py
this runs the main loop and holds the settings for the simulation. 


'''

from email import parser
from email.mime import image
import sys
from turtle import left
from urllib import response
import os



# # os.chdir('./datagen')

# sys.path.append('../')
# sys.path.append('../../')
# sys.path.append('../../../')

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
import argparse
import os
import time
from math import *
import time

import cv2
from copy import deepcopy

# os.chdir('../')
# print(os.getcwd())
# import imitation.ResNet8 as resnet8
# import models.racenet8 as racenet8
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))
from ..models.racenet8 import RaceNet8
from  ..models.ResNet8 import ResNet8

# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
# from VelocityPID import VelocityPID
from UnityPID import VelocityPID

from util import *

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser('Add argument for AirsimClient')
parser.add_argument('--weight','-w',type=str,default='')
parser.add_argument('--architecture','-arc', type=str, choices=["resnet8", "racenet8"])
arg = parser.parse_args()
arc = arg.architecture
model_weight_path = arg.weight


class NetworkTestClient(SimClient):

    def __init__(self, modelPath, raceTrackName="track0", device='cpu', arc=''):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False)

        # do custom setup here

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        if arc == 'resnet8':
            self.model = ResNet8(input_dim=3, output_dim=4, f=.5)
        if arc == 'racenet8':
            self.model = RaceNet8(input_dim=3, output_dim=4, f=.5)
        
        # if device == 'cuda':
        #     self.model = nn.DataParallel(self.model)
        #     cudnn.benchmark = True

        self.model.load_state_dict(torch.load(modelPath))

        self.device = device
        self.dev = torch.device(device)
        # self.client = airsim.MultirotorClient()
        self.model.to(self.dev)
        self.model.eval()
    
    def run(self):

        self.client.simPause(False)

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()
            # self.client.simPause(True)
        self.client.rotateToYawAsync(90)
            # self.client.simPause(False)
        time.sleep(3)
        cstatetest = self.getState()
        print(f'ox = {cstatetest[0]}, oy = {cstatetest[1]} oz= {cstatetest[2]}')
        if mission:
            print("Mission started: ")
        

        lastImage = time.time()

        timePerImage = 1. / float(self.config.framerate)

        cimageindex = 0

        while mission:

            
            # get current time and time delta
            tn = time.time()

            nextImage = tn - lastImage > timePerImage

            # if cimageindex == 0:

            # print("Mission continue")

            if nextImage:
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)


                # get images from AirSim API
                res = self.client.simGetImages([
                    airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False)])
                img = res[0]
                img1d = np.fromstring(img.image_data_uint8, dtype=np.uint8)
                imgsave = img1d.reshape(img.height, img.width, 3)
                image = self.loadWithAirsim()
                # cv2.imwrite("/media/data2/teamICRA/X4Gates_Circles_rl18tracks/Onegate-l" , images)

                images = torch.unsqueeze(image, dim=0)
                images = images.to(self.dev)

                # predict vector with network
                pred = self.model(images)
                pred = pred.to(torch.device('cpu'))
                pred = pred.detach().numpy()
                print(pred)
                pred = pred[0]  # remove batch
                # print(type(imgsave))
                cv2.imwrite(f"/media/data2/teamICRA/X4Gates_Circles_rl18tracks/Onegate-l/test/image{cimageindex}.png" , imgsave)
                # print(f"x = {pred[0]})
                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                # pausedelta = postpause - prepause
                # print(pausedelta)
                # if self.config.debug:
                #     self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                # else:
                #     print(f"pausedelta: {pausedelta}")
                # tn += pausedelta
                # lastImage = tn

                # send control command to airsim
                cstate = self.getState()
                # print(f'x = {cstate[0]}, y = {cstate[1]} z= {cstate[2]}')
                # rotate velocity command such that it is in world coordinates
                # Wvel = vector_body_to_world(pred[:3]*10, [0, 0, 0], cstate[3])
                Wvel = pred[:3]
                # print (f'yaw radiant = {pred[3]}')
                Wyaw = degrees(pred[3])
                # add pid output for yaw to current yaw position
                # Wyaw = degrees(cstate[3]) - degrees(pred[3])

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
                scale = 1
                Wvel = Wvel* scale
                print(f"World velos: x={float(Wvel[0])}, y={float(Wvel[1])}, z = {float(Wvel[2])}, yaw={float(Wyaw)}")
                self.client.moveByVelocityAsync(float(Wvel[0]),float(Wvel[1]), float(Wvel[2]),
                                                duration=float(timePerImage), yaw_mode=airsim.YawMode(False, Wyaw+90))




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
            modelPath=model_weight_path,
            device="cuda", arc=arc)) as nc:
        nc.loadGatePositions(nc.config.gates['poses'])
        nc.run()
