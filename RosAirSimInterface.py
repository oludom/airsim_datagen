#!/usr/bin/env python

'''
ROS wrapper version of AirSimInterface.py
adds ROS node functionality ontop of the airsim controller

'''

from AirSimInterface import AirSimInterface


class RosAirSimInterface(AirSimInterface):

    def __init__(self):
        # do custom initialization here

        # init super class (AirSimController)
        super().__init__()
