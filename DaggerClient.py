from fileinput import close
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
import cv2
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
        self.border_with = 20

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
        if showMarkers:
                            self.client.simPlotPoints(Wpath, color_rgba=[1.0, 0.0, 0.0, 1.0], size=10.0, duration=-1.0,
                                                is_persistent=True)

        lastWP = time.time()
        lastImage = time.time()
        lastIMU = time.time()
        lastPID = time.time()
        lastBG = time.time()

        WLastAirSimVel = [0., 0., 0., 0.]

        timePerWP = float(self.config.roundtime) / len(WpathComplete)
        timePerImage = 1. / float(self.config.framerate)
        timePerIMU = 1. / float(self.config.imuRate)
        timePerPID = 1. / float(self.config.pidRate)
        timePerBG = 1. / float(self.config.backgroundChangeRate)

        cwpindex = 0
        cimageindex = 0

        if self.config.debug:
            self.c.clear()

        def angleDifference(a: float, b: float):
            return (a - b + 540) % 360 - 180

        action = None
        # start = time.time()
        gate_index = 1
        # controll loop
        temp_index = 1
        lastindex = 0
        number_generateWP = 0
        return_gate_index = 0
        
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
            nextBG = tn - lastBG > timePerBG
        
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

            if nextBG:
                self.changeBackground()
                lastBG = tn

            if nextIMU:
                self.captureIMU()
                lastIMU = tn
            Passnew_WP = True
            return_gate_index = 0
            if nextPID:
                # get current state
                Wcstate = self.getState()  # rad
                # print('check_temp_index-1', temp_index)
                # get index closes wp
                prepause = time.time()
                self.client.simPause(True)
                closest_wp = self.getClosestWP2UAV(WpathComplete)
                # print('check_closest_WP',closest_wp)
                
                if cwpindex > 0 and (cwpindex % self.config.waypoints_per_segment == 0):
                    gate_index = temp_index + 1
                    return_gate_index = 0
                gate_temp = gate_index
                # print('change gate index',gate_index
                number_generateWP = gate_index * 50 - cwpindex
                # print('number_generate_check case 1',number_generateWP)

                if closest_wp > gate_index * self.config.waypoints_per_segment and closest_wp > cwpindex: 
                    gate_index = gate_temp + 1
                    number_generateWP = gate_index * 50 - cwpindex
                    return_gate_index = 0
                    # print('Increase_gate_index')
                    # print('number_generate_check case 2',number_generateWP)
                elif closest_wp < (gate_index -1) * self.config.waypoints_per_segment:
                    return_gate_index = 1
                    gate_index = gate_temp - return_gate_index
                    # print('hold-gate-index')
                    number_generateWP = gate_index * 50 - closest_wp
                    # print('number_generate_check case 3',number_generateWP)
                    Passnew_WP = False
               
                if cimageindex % 10 ==0:
                    action = np.random.choice(self.actions, p=[self.beta, 1 - self.beta])
                    
                
                if action == "expert" and cimageindex % 50 == 0 and self.beta > 0.3:

                        WpathComplete = self.MavericLocalPlaner(gate_index ,number_generateWP, WpathComplete,cwpindex,  showMarkers)

                if action == "expert" and cimageindex % 20 == 0 and self.beta <= 0.3:

                        WpathComplete = self.MavericLocalPlaner(gate_index ,number_generateWP, WpathComplete,cwpindex,  showMarkers)
                        

                self.client.simPause(False)
                postpause = time.time()
                pausedelta = prepause -postpause
                lastWP += pausedelta
                tn += pausedelta

                wp = WpathComplete[cwpindex]       
                # set goal state of pid controller
                Bgoal = vector_world_to_body(wp[:3], Wcstate[:3], Wcstate[3])  # rad
                # desired yaw angle is target point yaw angle world minus current uav yaw angle world 
                ByawGoal = angleDifference(wp[3], degrees(Wcstate[3]))  # deg
                # print(f"angle target: {ByawGoal:5.4f}")
                ctrl.setGoal([*Bgoal, ByawGoal])
                # update pid controller
                ctrl.update(tn - lastPID)
                # get current pid output´
                Bvel, Byaw = ctrl.getVelocityYaw()  # deg

                # pause simulation
                prepause = time.time()
                self.client.simPause(True)

                # save images of current frame
                image, depthimage = self.captureAndSaveImages(cwpindex, cimageindex, [*Bvel, Byaw], WLastAirSimVel)

                sample = self.loadWithAirsim(image, depthimage, config.input_channels['depth'])

                cimageindex += 1

                # print(action)                
                if action == "agent":
                    print(action)
                    images = torch.unsqueeze(sample, dim=0)
                    images = images.to(self.dev)

                    # predict vector with network
                    s = now()
                    pred = self.model(images)
                    # pd(s, "inference")
                    pred = pred.to(torch.device('cpu'))
                    pred = pred.detach().numpy()
                    pred = pred[0]  # remove batch

                    Bvel, Byaw = pred[:3], pred[3]  # rad
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
            # return gate_index 
            if not Passnew_WP:
                gate_index = gate_temp 
            # increase current waypoint index if time per waypoint passed and if there are more waypoints available in path
            if nextWP and len(WpathComplete) > (cwpindex + 1) and Passnew_WP:
                print('current_wp', cwpindex)
                temp_index = gate_index
                cwpindex = cwpindex + 1
                lastWP = tn
            # end mission when no more waypoints available
            if len(WpathComplete) - 10 <= (cwpindex + 1):  # ignore last 80 waypoints
                mission = False
            # end = time.time()
            # if end - start >= 1:
            #     print(f'image_id={cimageindex}')
            #     start=time.time()
    def MavericLocalPlaner(self, gate_index ,number_generateWP, WpathComplete,cwpindex,  showMarkers = False):
        """ Local Planner with number of time use expert policy
                Args : 
                    gate_index : gate index frome 1 -> 6 ( in domain randomization setting)
                    number_geberateWp: number of waypoint to generate to the next gate
                    WpathComplete : Global path
                    cwpindex: current waypoint index
                Return:
                    WpathComplete : return Local path added to the Global path
                """
        
        Ltimed_waypoints, Ltrajectory = self.generateTrajectoryToNextGatePositions(gate_index,timestep=1)

        Lpath, LpathComplete = self.convertTrajectoryToNextWaypoints(Ltimed_waypoints, Ltrajectory, number_generateWP,
                                                                evaltime=self.config.roundtime)

        if showMarkers:
            self.client.simPlotPoints(Lpath, color_rgba=[1.0, 0.0, 1.0, 1.0], size=10.0, duration=-1.0,
                                is_persistent=True)
                                
        L = 0
        # print('check',cwpindex + (self.config.waypoints_per_segment / number_regenerate_per_segment))/
        if len(LpathComplete)  > number_generateWP + 6 and (cwpindex + number_generateWP) <= gate_index * self.config.waypoints_per_segment:
            for w in range(cwpindex, int(cwpindex + number_generateWP)):
                # print('Wwaypoint count',w)
                # print('LocalWP count', L)
                WpathComplete[w] = LpathComplete[L+5]
                L += 1
        return WpathComplete

    def loadWithAirsim(self, image, depthimage, withDepth=False):

        sample = None
        depth = None
        kp = None
        
        dim = (config.image_dim[1], config.image_dim[0])
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        if withDepth:
            # format depth image
            depth = pfm.get_pfm_array(depthimage)  # [0] ignores scale

        if config.input_channels['orb'] or config.input_channels['sparse']:
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
        if config.input_channels['sparse']:
            orbmask = torch.zeros_like(image[0], dtype=torch.bool)
            for el in kp:
                x, y = el.pt
                y = min(max(int(y), self.border_with//2), image.shape[1]-self.border_with//2)
                x = min(max(int(x), self.border_with//2), image.shape[2]-self.border_with//2)
                # orb_box  = image[:, y-5:y+5, x-5:x+5]
                orbmask[y-self.border_with//2:y+self.border_with//2, x-self.border_with//2:x+self.border_with//2] =1
            image[:, ~orbmask] = 0
            sample = image
        return sample
