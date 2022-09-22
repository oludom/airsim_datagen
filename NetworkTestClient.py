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

    def __init__(self, modelPath, raceTrackName="track0", device='cpu', *args, **kwargs):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False, *args, **kwargs)

        # do custom setup here

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        # dataset_basepath = "/media/micha/eSSD/datasets"
        dataset_basepath = self.config.dataset_basepath
        # dataset_basepath = "/data/datasets"
        # dataset_basename = "X4Gates_Circle_right_"
        dataset_basename = self.DATASET_NAME
        # dataset_basename = "X4Gates_Circle_2"

        self.trajectoryFile = f'{dataset_basepath}/{dataset_basename}/{raceTrackName}/trajectory_{config.itypes}.tum'


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

        # set backgorund texture
        self.changeBackgroundTest()

        lastImage = time.time()

        timePerImage = 1. / float(self.config.framerate)

        cimageindex = 0

        mission_start = now()

        # trajectory file
        with open(self.trajectoryFile, 'w') as traj:

            while mission:

                # get current time and time delta
                tn = time.time()

                nextImage = tn - lastImage > timePerImage

                if nextImage:
                    # pause simulation
                    prepause = time.time()
                    self.client.simPause(True)


                    # get images from AirSim API

                    image, segres = self.loadWithAirsim(config.input_channels['depth'])

                    if not self.checkGateInView(segres):
                        mission = False 
                        pd(mission_start, "gates out of view after")
                        return

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
                    mission_start += pausedelta

                    Bvel = pred[:3]

                    # print(f"magnitude: {magnitude(Bvel)}")
                    Bvel_percent = magnitude(Bvel) / 2
                    # print(f"percent: {Bvel_percent*100}")
                    # if magnitude of pid output is greater than velocity limit, scale pid output to velocity limit
                    # if Bvel_percent > 1:
                    Bvel = Bvel / Bvel_percent
                    Byaw = pred[3] / Bvel_percent

                    # print(f"y: {degrees(Byaw)}")

                    ypercent = abs(degrees(Byaw) / 10)
                    # print(f"ypercent: {ypercent}")
                    if ypercent > 1:
                        # print("limiting yaw") 
                        Bvel = Bvel / ypercent
                        Byaw = Byaw / ypercent

                    # send control command to airsim
                    cstate = self.getState()

                    # rotate velocity command such that it is in world coordinates
                    Wvel = vector_body_to_world(Bvel, [0, 0, 0], cstate[3])

                    # add pid output for yaw to current yaw position
                    Wyaw = degrees(cstate[3]) + degrees(Byaw)

                    # save current pose
                    pose = self.getPositionUAV()
                    traj.write(" ".join([str(el) for el in [tn, *pose]]) + "\n")


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
                    
                





    def loadWithAirsim(self, withDepth = False):
        
        start = now()
        # AirSim API rarely returns empty image data
        # 'and True' emulates a do while loop
        loopcount = 0
        sample = None
        segres = None
        while (True):
            if withDepth:

                # get images from AirSim API
                res = self.client.simGetImages(
                    [
                        airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                        # airsim.ImageRequest("front_right", airsim.ImageType.Scene),
                        airsim.ImageRequest("depth_cam", airsim.ImageType.DepthPlanar, True),
                        airsim.ImageRequest("seg", airsim.ImageType.Segmentation, False, False)
                    ]
                )
                segres = res[2]
            else:
                res = self.client.simGetImages(
                    [
                        airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest("seg", airsim.ImageType.Segmentation, False, False)
                    ]
                )
                segres = res[1]
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

        return sample, segres


if __name__ == "__main__":
    import contextlib

    for i in range(10):
        track = "track"+str(i)
        print(f"current track: {track}")

        with contextlib.closing(NetworkTestClient(
                f"/home/kristoffer/dev/orb_imitation/datagen/eval/runs/domain_randomization/ResNet8_ds=dr_pretrain_l={config.itypes}_f=0.5_bs=32_lt=MSE_lr=0.001_c=run0/epoch7.pth",
                device=config.device, raceTrackName=track, configFilePath='config/config_dr_test.json')) as nc:
            # nc.loadGatePositions([[5.055624961853027, -0.7640624642372131+4, -0.75, -90.0]])
            nc.run(uav_position=nc.config.uav_position)
