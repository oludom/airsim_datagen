#!/usr/bin/env python

'''

simulation client for AirSimInterface.py
this runs the main loop and holds the settings for the simulation. 


'''

from AirSimInterface import AirSimInterface

import airsim
import numpy as np
import pprint
import curses

import os
import time
from math import *
import time

import cv2
from copy import deepcopy

from util import *

# import MAVeric polynomial trajectory planner
import MAVeric.trajectory_planner as maveric

# use custom PID controller
# from VelocityPID import VelocityPID
from UnityPID import VelocityPID


class SimClient(AirSimInterface):

    def __init__(self, raceTrackName="track0", *args, **kwargs):

        # init super class (AirSimController)
        super().__init__(raceTrackName=raceTrackName, *args, **kwargs)

        # do custom setup here

        if self.config.debug:
            self.c = curses.initscr()
            curses.noecho()
            curses.cbreak()

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        self.timestep = 1. / self.config.framerate

    '''
    this function should be called at end of lifetime of this object (see contextlib)
    close all opened files here
    and reset simulation
    '''

    def close(self):

        self.client.simPause(False)

        if self.config.debug:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        self.client.simFlushPersistentMarkers()

        # close super class (AirSimController)
        super().close()

    '''
    run mission
    - initialize velocity pid controller
    - generate trajectory through gates
    - save current configuration to dataset folder
    - follow trajectory with sampled waypoints
    showMarkers: boolean, if true, trajectory will be visualized with red markers in simulation
    captureImages: boolean, if true, each iteration will capture a frame of each camera, simulation is paused for this

    variable name prefix:
    W: coordinates in world frame
    B: coordinates in body frame (drone/uav)
    '''

    def gateMission(self, showMarkers=True, captureImages=True):

        mission = True

        # reset sim
        self.reset()

        # takeoff
        self.client.takeoffAsync().join()

        # make sure drone is not drifting anymore after takeoff
        time.sleep(3)

        # init pid controller for velocity control
        pp = 2
        dd = .2
        ii = .0

        Kp = np.array([pp, pp, pp])
        Ki = np.array([ii, ii, ii])
        Kd = np.array([dd, dd, dd])
        yaw_gain = np.array([.2, 0., 25])  # [.25, 0, 0.25]

        distance_threshold = 0.01
        angle_threshold = 0.1

        ctrl = VelocityPID(Kp, Ki, Kd, yaw_gain, distance_threshold, angle_threshold)

        # set initial state and goal to 0
        ctrl.setState([0, 0, 0, 0])
        ctrl.setGoal([0, 0, 0, 0])

        # get trajectory
        Wtimed_waypoints, Wtrajectory = self.generateTrajectoryFromCurrentGatePositions(timestep=1)

        Wpath, WpathComplete = self.convertTrajectoryToWaypoints(Wtimed_waypoints, Wtrajectory,
                                                                 evaltime=self.config.roundtime)

        # save current configuration and trajectory in data set folder
        data = {
            "waypoints": WpathComplete
        }
        self.saveConfigToDataset(gateConfig, data)

        # show trajectory
        if showMarkers:
            self.client.simPlotPoints(Wpath, color_rgba=[1.0, 0.0, 0.0, .2], size=10.0, duration=-1.0,
                                      is_persistent=True)

        lastWP = time.time()
        lastImage = time.time()
        lastIMU = time.time()
        lastPID = time.time()

        timePerWP = float(self.config.roundtime) / len(WpathComplete)
        timePerImage = 1. / float(self.config.framerate)
        timePerIMU = 1. / float(self.config.imuRate)
        timePerPID = 1. / float(self.config.pidRate)

        cwpindex = 0
        cimageindex = 0
        if self.config.debug:
            self.c.clear()

        def angleDifference(a: float, b: float):
            return (a - b + 540) % 360 - 180

        # controll loop
        while mission:

            # if self.config.debug:
            #     self.c.clear()

            # get and plot current waypoint (blue)
            wp = WpathComplete[cwpindex]

            # show markers if applicable
            self.showMarkers(showMarkers, wp)

            # get current time and time delta
            tn = time.time()

            nextWP = tn - lastWP > timePerWP
            nextImage = tn - lastImage > timePerImage
            nextIMU = tn - lastIMU > timePerIMU
            nextPID = tn - lastPID > timePerPID

            if showMarkers:
                current_drone_pose = self.getPositionUAV()
                self.client.simPlotPoints(
                    [airsim.Vector3r(current_drone_pose[0], current_drone_pose[1], current_drone_pose[2])],
                    color_rgba=[1.0, 0.0, 1.0, 1.0],
                    size=10.0, duration=self.timestep, is_persistent=False)

                self.client.simPlotPoints(
                    [airsim.Vector3r(current_drone_pose[0], current_drone_pose[1], current_drone_pose[2])],
                    color_rgba=[1.0, 0.6, 1.0, .5],
                    size=10.0, duration=self.timestep, is_persistent=True)

            if self.config.debug:
                if nextWP:
                    self.c.addstr(3, 0, f"wpt: {format(1. / float(tn - lastWP), '.4f')}hz")
                if nextImage:
                    self.c.addstr(4, 0, f"img: {format(1. / float(tn - lastImage), '.4f')}hz")
                if nextIMU:
                    self.c.addstr(5, 0, f"imu: {format(1. / float(tn - lastIMU), '.4f')}hz")
                if nextPID:
                    self.c.addstr(6, 0, f"pid: {format(1. / float(tn - lastPID), '.4f')}hz")

            if nextIMU:
                self.captureIMU()
                lastIMU = tn

            if nextImage and captureImages:
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)

                Bvel, Byaw = ctrl.getVelocityYaw()

                # save images of current frame
                self.captureAndSaveImages(cwpindex, cimageindex, [*Bvel, Byaw])
                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                if self.config.debug:
                    self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
                lastWP += pausedelta
                lastIMU += pausedelta
                tn += pausedelta
                lastImage = tn

            if nextPID:
                # get current state
                Wcstate = self.getState()

                # set goal state of pid controller
                Bgoal = vector_world_to_body(wp[:3], Wcstate[:3], Wcstate[3])
                # desired yaw angle is target point yaw angle world minus current uav yaw angle world 
                ByawGoal = angleDifference(wp[3], degrees(Wcstate[3]))
                print(f"angle target: {ByawGoal:5.4f}")
                ctrl.setGoal([*Bgoal, ByawGoal])
                # update pid controller
                ctrl.update(tn - lastPID)
                # get current pid output´
                Bvel, Byaw = ctrl.getVelocityYaw()

                # rotate velocity command such that it is in world coordinates
                Wvel = vector_body_to_world(Bvel, [0, 0, 0], Wcstate[3])

                # add pid output for yaw to current yaw position
                Wyaw = degrees(Wcstate[3]) + Byaw

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
                                                duration=float(timePerPID), yaw_mode=airsim.YawMode(False, Wyaw))

                # save last PID time
                lastPID = tn

            # debug output
            if self.config.debug:
                self.c.refresh()

            # increase current waypoint index if time per waypoint passed and if there are more waypoints available in path
            if nextWP and len(WpathComplete) > (cwpindex + 1):
                cwpindex = cwpindex + 1
                lastWP = tn
            # end mission when no more waypoints available
            if len(WpathComplete) <= (cwpindex + 1):
                mission = False
        if showMarkers:
            # clear persistent markers
            self.client.simFlushPersistentMarkers()

    def showMarkers(self, showMarkers, wp):
        if showMarkers:
            self.client.simPlotPoints([airsim.Vector3r(wp[0], wp[1], wp[2])], color_rgba=[0.0, 0.0, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)

    # set gate poses in simulation to provided configuration
    # gates: [ [x, y, z, yaw], ...] 
    def loadGatePositions(self, gates):
        # load gate positions
        for i, gate in enumerate(gates):
            self.setPositionGate(i + 1, gate)

    # load the next gate pose configuration in self.gateConfigurations
    # if index overflows, wraps around to 0 and repeats configurations
    def loadNextGatePosition(self):
        currentConfiguration = self.gateConfigurations[self.currentGateConfiguration]
        self.loadGatePositions(currentConfiguration)

        self.currentGateConfiguration += 1
        if self.currentGateConfiguration >= len(self.gateConfigurations):
            self.currentGateConfiguration = 0

    '''
    generate random gate pose configurations
    '''

    def generateGateConfigurations(self):
        # reset fields
        self.currentGateConfiguration = 0
        self.gateConfigurations = []
        gridsize = self.config.gates["grid_size"]

        # generate range for each axis
        xr0 = self.config.gates["range"]["x"][0]
        xr1 = self.config.gates["range"]["x"][1]
        xr = list(np.linspace(xr0, xr1, gridsize))

        yr0 = self.config.gates["range"]["y"][0]
        yr1 = self.config.gates["range"]["y"][1]
        yr = list(np.linspace(yr0, yr1, gridsize))

        zr0 = self.config.gates["range"]["z"][0]
        zr1 = self.config.gates["range"]["z"][1]
        zr = list(np.linspace(zr0, zr1, gridsize))

        wr0 = self.config.gates["range"]["yaw"][0]
        wr1 = self.config.gates["range"]["yaw"][1]
        wr = list(np.linspace(wr0, wr1, gridsize))

        gateCenters = self.config.gates['poses']

        # generate configurations
        for i in range(self.config.gates['number_configurations']):
            currentConfiguration = []
            for gate in gateCenters:
                # create new gate from gate center
                newGate = np.array(deepcopy(gate))
                ra = np.random.randint(gridsize, size=4)
                # move gate by random offset
                newGate += np.array([xr[ra[0]], yr[ra[1]], zr[ra[2]], wr[ra[3]]])
                currentConfiguration.append(newGate)

            # append configuration
            self.gateConfigurations.append(currentConfiguration)

        # remove duplicates
        self.gateConfigurations = np.array(self.gateConfigurations)
        remove_duplicates = lambda data: np.unique([tuple(row) for row in data], axis=0)
        self.gateConfigurations = remove_duplicates(self.gateConfigurations)
        self.gateConfigurations.tolist()

    # helper function for visualizing gate center points and waypoints offset from gate centers with certain distance from gate and same rotation, unused
    def printGateWPs(self):

        self.reset()
        self.client.takeoffAsync().join()
        self.loadGatePositions(self.config.gates['poses'])

        self.captureAndSaveImages()

        gates = [
            [5.055624961853027, -0.7640624642372131, -0.75],
            [10.555624961853027, 3.7359328269958496, -0.75],
            [5.055624961853027, 8.235932350158691, -0.75],
            [1.0556249618530273, 3.7359390258789062, -0.75]
        ]

        wps = self.generateTrajectoryFromCurrentGatePositions(1, False)
        plot = [self.toVector3r(wp) for wp in wps]
        self.client.simPlotPoints(plot, color_rgba=[1.0, 0.0, 0.0, .2], size=10.0, duration=-1, is_persistent=True)

        plot = [self.toVector3r(wp) for wp in gates]
        self.client.simPlotPoints(plot, color_rgba=[0.0, 1.0, 0.0, .2], size=10.0, duration=-1, is_persistent=True)

        print("wps")
        for wp in wps:
            print(wp)

        time.sleep(10)
        self.client.simFlushPersistentMarkers()

    # test to check roration of aditional waypoints for each gate, unused
    def rotationTest(self):

        self.reset()
        self.client.takeoffAsync().join()
        self.loadGatePositions(self.config.gates['poses'])

        yaw = radians(90)
        gate = np.array([5.055624961853027, -0.7640624642372131, -0.75])

        gate1 = deepcopy(gate)

        y1 = np.array([0, 1, 0])
        # rotation matrix z-axis
        rot = np.array([[cos(yaw), sin(yaw), 0], [-1 * sin(yaw), cos(yaw), 0], [0, 0, 1]])
        rot1 = rot.dot(np.transpose(y1))
        gate1 += rot1

        # test vector
        tv = np.array([1, 0, 0])

        tvout = rot.dot(np.transpose(tv))
        print("tv", tv)
        print("to", tvout)

        gate2 = deepcopy(gate)
        gate2 -= rot1

        plot = [self.toVector3r(wp) for wp in [gate1, gate2]]

        self.client.simPlotPoints(plot, color_rgba=[1.0, 0.0, 0.0, .2], size=10.0, duration=-1, is_persistent=True)
        self.client.simPlotPoints([self.toVector3r(gate)], color_rgba=[0.0, 1.0, 0.0, .2], size=10.0, duration=-1,
                                  is_persistent=True)
        time.sleep(10)
        self.client.simFlushPersistentMarkers()

    # wp: [x, y, z]
    # returns airsim.Vector3r()
    def toVector3r(self, wp):
        return airsim.Vector3r(wp[0], wp[1], wp[2])


if __name__ == "__main__":

    import contextlib

    configurations = []

    with contextlib.closing(SimClient()) as sc:
        # generate random gate configurations within bounds set in config.json
        sc.generateGateConfigurations()
        configurations = deepcopy(sc.gateConfigurations)

    for i, gateConfig in enumerate(configurations):
        with contextlib.closing(SimClient(raceTrackName=f"track{i}")) as sc:
            sc.gateConfigurations = [gateConfig]

            sc.loadNextGatePosition()

            # fly mission
            sc.gateMission(False, True)

            sc.loadGatePositions(sc.config.gates['poses'])
            sc.reset()
