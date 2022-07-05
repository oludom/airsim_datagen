#!/usr/bin/env python

'''

simulation client for AirSimController.py 
this runs the main loop and holds the settings for the simulation. 


'''

from AirSimController import AirSimController
from SimClient import SimClient

import airsim
import numpy as np
import pprint
import curses
import torch

import os
import time
from math import *
import time

import cv2
from copy import deepcopy


# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
# from VelocityPID import VelocityPID
from UnityPID import VelocityPID


class NetworkTestClient(SimClient):

    def __init__(self, raceTrackName="track0"):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, createDataset=False)

        # do custom setup here


        self.gateConfigurations = []
        self.currentGateConfiguration = 0

if __name__ == "__main__":

    import contextlib

    with contextlib.closing(NetworkTestClient()) as nc:
        pass