#!/usr/bin/env python

'''

simulation client for AirSimInterface.py
this runs the main loop and holds the settings for the simulation. 


'''
import pfm

from AirSimInterface import AirSimInterface
from SimClient import SimClient

import airsim
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn

import time
from math import *
import time

from copy import deepcopy

import ResNet8 as resnet8
import orb

# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
# from VelocityPID import VelocityPID
from UnityPID import VelocityPID

from util import *
import config

torch.set_grad_enabled(False)


now = lambda: int(round(time.time() * 1000))
pd = lambda s, t: print(f"{t}: {now() - s}ms")

class NetworkTestClient(SimClient):

    def __init__(self, modelPath, raceTrackName="track0", device='cpu'):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False)

        # do custom setup here

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        # dataset_basepath = "/media/micha/eSSD/datasets"
        dataset_basepath = self.config.dataset_basepath
        # dataset_basepath = "/data/datasets"
        # dataset_basename = "X4Gates_Circle_right_"
        dataset_basename = self.DATASET_NAME
        # dataset_basename = "X4Gates_Circle_2"


        # relead config file from dataset
        # configuration file
        self.configFile = open(f'{dataset_basepath}/{dataset_basename}/{raceTrackName}/config.json', "r")

        self.config = {}
        self.loadConfig(self.configFile)

        self.loadGatePositions(self.config.gates['poses'])

        self.model = resnet8.ResNet8(input_dim=config.num_input_channels, output_dim=4, f=config.resnet_factor)
        if device == 'cuda':
            self.model = nn.DataParallel(self.model)
            cudnn.benchmark = True

        self.model.load_state_dict(torch.load(modelPath))

        self.device = device
        self.dev = torch.device(device)

        self.model.to(self.dev)
        self.model.eval()

    def run(self, uav_position=None):

        self.client.simPause(False)

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        if uav_position:
            uav_position[3] = uav_position[3] + 90
            self.setPositionUAV(uav_position)
            self.client.moveByVelocityAsync(float(0), float(0), float(0),
                                                duration=float(3), yaw_mode=airsim.YawMode(False, uav_position[3]))

        time.sleep(3)

        lastImage = time.time()

        timePerImage = 1. / float(self.config.framerate)

        cimageindex = 0

        mission_start = now()

        while mission:

            # get current time and time delta
            tn = time.time()

            nextImage = tn - lastImage > timePerImage

            if nextImage:
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)


                # get images from AirSim API

                image = self.loadWithAirsim(config.input_channels['depth'])

                images = torch.unsqueeze(image, dim=0)
                images = images.to(self.dev)

                # predict vector with network
                s = now()
                pred = self.model(images)
                # pd(s, "inference")
                pred = pred.to(torch.device('cpu'))
                pred = pred.detach().numpy()
                pred = pred[0]  # remove batch

                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                if self.config.debug:
                    self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                # else:
                #     print(f"pausedelta: {pausedelta}")
                tn += pausedelta
                lastImage = tn

                # send control command to airsim
                cstate = self.getState()

                # rotate velocity command such that it is in world coordinates
                Wvel = vector_body_to_world(pred[:3]*2, [0, 0, 0], cstate[3])

                # add pid output for yaw to current yaw position
                Wyaw = degrees(cstate[3]) + degrees(pred[3])

                # visualizes prediction 
                # self.client.simPlotPoints([self.getPositionAirsimUAV().position], color_rgba=[1.0, 0.0, 1.0, 1.0],
                #                       size=10.0, duration=self.timestep, is_persistent=False)
                # Wposvel = cstate[:3] + Wvel
                # self.client.simPlotPoints([airsim.Vector3r(Wposvel[0], Wposvel[1], Wposvel[2])], color_rgba=[.8, 0.5, 1.0, 1.0],
                #                       size=10.0, duration=self.timestep, is_persistent=False)

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
                
                
                # if now() - mission_start > 50000:
                #     mission = False
                #     pd(mission_start, "end of mission after")
                #     return





    def loadWithAirsim(self, withDepth = False):
        
        start = now()
        # AirSim API rarely returns empty image data
        # 'and True' emulates a do while loop
        loopcount = 0
        sample = None
        while (True):
            if withDepth:

                # get images from AirSim API
                res = self.client.simGetImages(
                    [
                        airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                        # airsim.ImageRequest("front_right", airsim.ImageType.Scene),
                        airsim.ImageRequest("depth_cam", airsim.ImageType.DepthPlanar, True)
                    ]
                )
            else:
                res = self.client.simGetImages(
                    [
                        airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False)
                    ]
                )
            left = res[0]
            # pd(start, f"lc{loopcount}")

            img1d = np.fromstring(left.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(left.height, left.width, 3)

            # pd(start, f"s1")

            # check if image contains data, repeat request if empty
            if image.size:
                break  # end of do while loop
            else:
                loopcount += 1
                print("airsim returned empty image." + str(loopcount))

        if withDepth:
            # format depth image
            depth = pfm.get_pfm_array(res[1]) # [0] ignores scale
            # pd(start, f"d2")

        if config.input_channels['orb']:
            kp, des, _, _, _ = orb.get_orb(image)
            # pd(start, f"o3")

        # preprocess image
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        if config.input_channels['rgb']:
            sample = image

        # pd(start, f"i4")

        if withDepth:
            depth = transforms.Compose([
                transforms.ToTensor(),
            ])(depth)
            # pd(start, f"d5")
            if sample is not None:
                sample = torch.cat((sample, depth), dim=0)
            else:
                sample = depth
            # pd(start, f"d6")

        if config.tf:
            sample = config.tf(sample)
            # pd(start, f"tf7")

        if config.input_channels['orb']:
            orbmask = torch.zeros_like(image[0])
            for el in kp:
                x, y = el.pt
                orbmask[int(y), int(x)] = 1
            orbmask = orbmask.unsqueeze(0)
            if sample is not None:
                sample = torch.cat((sample, orbmask), 0)
            else:
                sample = orbmask

        return sample


if __name__ == "__main__":
    import contextlib

    # for i in range(8):
    #     track = "track"+str(i)
    #     print(f"current track: {track}")
    track = "track6"

    with contextlib.closing(NetworkTestClient(
            "/home/kristoffer/dev/orb_imitation/datagen/eval/runs/X1Gate_evaluation/ResNet8_ds=X1Gate8tracks_l=do_f=0.25_bs=32_lt=MSE_lr=0.001_c=run0/epoch6.pth",
            device=config.device, raceTrackName=track)) as nc:
        # nc.loadGatePositions([[5.055624961853027, -0.7640624642372131+4, -0.75, -90.0]])
        nc.run(uav_position=nc.config.uav_position)
