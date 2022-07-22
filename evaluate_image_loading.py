#!/usr/bin/env python


'''
check weather loading the images different ways results in the same torch.Tensor
'''


import sys
from urllib import response

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from SimClient import SimClient

import airsim
import numpy as np
import pprint
import curses
import torch
import torchvision.transforms as transforms

import os
import time
from math import *
import time

import cv2
from copy import deepcopy

import imitation.dn as dn

from RaceTrackLoader import RaceTracksDataset




class Test(SimClient):

    def __init__(self, raceTrackName="track0"):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=True)


    
    def run(self):


        self.client.simPause(False)

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        time.sleep(3)

        self.client.simPause(True)

        # load images

        # export airsim
        self.captureAndSaveImages(0,0)

        # load with airsim
        image1 = self.loadWithAirsim()

        self.outputFile.flush()
        l = self.outputFile.tell()
        self.outputFile.seek(l-2)

        print(']\n}', file=self.outputFile)
        self.outputFile.close()

        data = {
            "waypoints": [[0,0,0,0,0]]
        }
        self.saveConfigToDataset(np.array([]), data)

        # load with torch dataloader
        image2 = self.loadWithTorch()


        i1 = image1.tolist()
        i2 = image2.tolist()

        print("end")


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

        image = dn.preprocess(image)
        return image


    def loadWithTorch(self):

        dataset = torch.utils.data.DataLoader(
            RaceTracksDataset(
                self.config.dataset_basepath,
                self.DATASET_NAME,
                device='cpu',
                maxTracksLoaded=35,
                imageScale=100,
                skipTracks=0,
                grayScale=False,
                imageTransforms=dn.preprocess
            ),
            batch_size=1,
            shuffle=True
        )
        batch = next(iter(dataset))
        image = batch[0][0]
        return image

        

if __name__ == "__main__":

    Test().run()

    import contextlib

    # with contextlib.closing(Test()) as nc:
    #     nc.run()