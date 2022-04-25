#!/usr/bin/env python

'''
ROS wrapper version of AirSimController.py
adds ROS node functionality ontop of the airsim controller

'''

from AirSimController import AirSimController

class RosAirSimController(AirSimController):

    def __init__(self):

        # do custom initialization here

        # init super class (AirSimController)
        super().__init__()