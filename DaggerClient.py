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


class DaggerClient(SimClient):

    def __init__(self, model, raceTrackName="track0", device='cpu', beta=0.99, *args, **kwargs):

        # init super class (SimClient)
        super().__init__(raceTrackName=raceTrackName, *args, **kwargs)

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        self.device = device
        self.dev = torch.device(device)

        self.model = model
        self.model.eval()

        self.beta = beta
        self.expert_steps = int(50 * self.beta)
        self.agent_Steps = 50 - self.expert_steps
        self.actions = np.array(['expert', 'agent'])

    def run(self, uav_position=None, showMarkers=False, velocity_limit=2.0):

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
            print('init')
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
        self.saveConfigToDataset(self.gateConfigurations[self.currentGateConfiguration], data)

        lastWP = time.time()
        lastImage = time.time()
        lastIMU = time.time()
        lastPID = time.time()

        WLastAirSimVel = [0., 0., 0., 0.]

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

        while mission:

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

            if nextPID:
                # get current state
                Wcstate = self.getState()

                # set goal state of pid controller
                Bgoal = vector_world_to_body(wp[:3], Wcstate[:3], Wcstate[3])
                # desired yaw angle is target point yaw angle world minus current uav yaw angle world 
                ByawGoal = angleDifference(wp[3], degrees(Wcstate[3]))
                # print(f"angle target: {ByawGoal:5.4f}")
                ctrl.setGoal([*Bgoal, ByawGoal])
                # update pid controller
                ctrl.update(tn - lastPID)
                # get current pid output´
                Bvel, Byaw = ctrl.getVelocityYaw()
                init_yaw = degrees(Wcstate[3]) + Byaw
                if cwpindex <= 1:
                    print(cwpindex)
                    self.client.moveByVelocityAsync(float(0), float(0), float(0),
                                            duration=float(3), yaw_mode=airsim.YawMode(False, init_yaw))
                print(cwpindex)
                
                # pause simulation
                prepause = time.time()
                self.client.simPause(True)

                # save images of current frame
                image, depthimage = self.captureAndSaveImages(cwpindex, cimageindex, [*Bvel, Byaw], WLastAirSimVel)

                sample = self.loadWithAirsim(image, depthimage, config.input_channels['depth'])

                cimageindex += 1

                action = np.random.choice(self.actions, p=[self.beta, 1 - self.beta])
                if action == "agent":
                    images = torch.unsqueeze(sample, dim=0)
                    images = images.to(self.dev)
                    
                    # predict vector with network
                    s = now()
                    pred = self.model(images)
                    # pd(s, "inference")
                    pred = pred.to(torch.device('cpu'))
                    pred = pred.detach().numpy()
                    pred = pred[0]  # remove batch

                    Bvel, Byaw = pred[:3], pred[3]
                    Byaw = degrees(Byaw)
                # limit magnitude to max velocity
                Bvel_percent = magnitude(Bvel) / velocity_limit
                # print(f"percent: {Bvel_percent*100}")
                # if magnitude of pid output is greater than velocity limit, scale pid output to velocity limit
                if Bvel_percent > 1:
                    Bvel = Bvel / Bvel_percent

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

                # rotate velocity command such that it is in world coordinates
                Wvel = vector_body_to_world(Bvel, [0, 0, 0], Wcstate[3])
                print(Byaw)
                # add pid output for yaw to current yaw position
                Wyaw = degrees(Wcstate[3]) + Byaw
                WLastAirSimVel = [*Wvel, Wyaw]

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
            if len(WpathComplete) - 50 <= (cwpindex + 1):  # ignore last 80 waypoints
                mission = False

    def loadWithAirsim(self, image, depthimage, withDepth=False):

        sample = None
        depth = None
        kp = None

        if withDepth:
            # format depth image
            depth = pfm.get_pfm_array(depthimage)  # [0] ignores scale

        if config.input_channels['orb']:
            kp, des, _, _, _ = orb.get_orb(image)

        # preprocess image
        image = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        if config.input_channels['rgb']:
            sample = image

        if withDepth:
            depth = transforms.Compose([
                transforms.ToTensor(),
            ])(depth)
            if sample is not None:
                sample = torch.cat((sample, depth), dim=0)
            else:
                sample = depth

        if config.tf:
            sample = config.tf(sample)

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
