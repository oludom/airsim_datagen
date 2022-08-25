from operator import mod
from re import I
from signal import pause
from tokenize import Triple
from SimClient import SimClient
from NetworkTestClient import NetworkTestClient
import time
import numpy as np
from UnityPID import VelocityPID
from copy import deepcopy
import airsim
import torch
import argparse
from AirSimInterface import AirSimInterface
from util import *
from math import *
import os
from models.ResNet8 import ResNet8
from models.racenet8 import RaceNet8
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import curses


parser = argparse.ArgumentParser('Add argument for AirsimClient')
parser.add_argument('--weight','-w',type=str,default='')
parser.add_argument('--track','-t',type=str,default='track0')
arg = parser.parse_args()
model_weight_path = arg.weight
track_name = arg.track
class DaggerClient(AirSimInterface):
    # def __init__(self, raceTrackName="track0", beta=0.99, *args, **kwargs):
    #     super().__init__(raceTrackName, *args, **kwargs)
    #     self.beta = beta
    #     self.expert_steps = int(50*beta)
    #     self.agent_Steps = 50 - self.expert_steps

    
    def __init__(self, raceTrackName=track_name, modelPath=None, device='cpu', beta=0.99, *args, **kwargs):
        super().__init__(raceTrackName, modelPath, *args, **kwargs)
        self.beta = beta
        self.expert_steps = int(50*self.beta)
        self.agent_Steps = 50 - self.expert_steps
        # self.createDataset = True
        # self.createDataset = createDataset
        if self.config.debug:
            self.c = curses.initscr()
            curses.noecho()
            curses.cbreak()

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        self.timestep = 1. / self.config.framerate
        self.model = ResNet8(input_dim=3, output_dim=4, f=1)
        print(modelPath)
        if modelPath is not None:
            self.model.load_state_dict(torch.load(modelPath))
            if device == 'cuda':
                self.model = nn.DataParallel(self.model)
                cudnn.benchmark = True

            # self.model.load_state_dict(torch.load(modelPath))

            self.device = device
            self.dev = torch.device(device)
            self.model.to(self.dev)
            self.model.eval()
    def close(self):

        self.client.simPause(False)

        if self.config.debug:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        self.client.simFlushPersistentMarkers()

        # close super class (AirSimController)
        super().close()
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
        # self.saveConfigToDataset(gateConfig, data)

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

            expert_step = cwpindex
            agent_step = 0
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
                print("Dung-------")
                # save images of current frame
                self.captureAndSaveImages(cwpindex, cimageindex, [*Bvel, Byaw])
                cimageindex += 1

                # unpause simulation
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = postpause - prepause
                # if self.config.debug:
                #     self.c.addstr(10, 0, f"pausedelta: {pausedelta}")
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
                # get current pid output
                # viet lai
                prepause = time.time()
                self.client.simPause(True)
                if (expert_step % self.expert_steps == 0) and (expert_step != 0): # agent policy 
                    # Agent Policy
                    image = self.loadWithAirsim()
                    images = torch.unsqueeze(image, dim=0)
                    images = images.to(self.dev)
                    
                    # predict vector with network
                    pred = self.model(images)
                    pred = pred.to(torch.device('cpu'))
                    pred = pred.detach().numpy()
                    pred = pred[0]  # remove batch
                    Bvel, Byaw = pred[:3], pred[3]
                    agent_step += 1
                    expert_step +=1
                    
                    # agent_step = 0

                    # rotate velocity command such that it is in world coordinates
                else: # Expert Policy 
                    Bvel, Byaw = ctrl.getVelocityYaw()
                    expert_step +=1
                self.client.simPause(False)
                postpause = time.time()
                pausedelta = prepause - postpause
                # tn += pausedelta
                # lastImage = tn
                # if (agent_step % self.agent_Steps == 0) and (agent_step != 0):
                #     agent_step = 0
                #     expert_step = +1 



                Wvel = vector_body_to_world(Bvel, [0, 0, 0], Wcstate[3])
                # add pid output for yaw to current yaw position
                Wyaw = degrees(Wcstate[3]) + Byaw

                '''
                Args:
                    vx (float): desired velocity in world (NED) X axis
                    vy (float): desired velocity icwpName of the multirotor to send this command to
                '''
                self.client.moveByVelocityAsync(float(Wvel[0]), float(Wvel[1]), float(Wvel[2]),
                                                duration=float(timePerPID), yaw_mode=airsim.YawMode(False, Wyaw))

                # save last PID time
                lastPID = tn + pausedelta

            # debug output
            if self.config.debug:
                self.c.refresh()

            # increase current waypoint index if time per waypoint passed and if there are more waypoints available in path
            print(nextWP)
            print(cwpindex)
            print(len(WpathComplete))
            
            if cwpindex == 130:
                mission = False
                break
            if nextWP and len(WpathComplete) > (cwpindex + 1):
                cwpindex = cwpindex + 1
                lastWP = tn

            # end mission when no more waypoints available
            # if len(WpathComplete) <= (cwpindex + 1):
            
        #### Done

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

    # wp: [x, y, z]
    # returns airsim.Vector3r()
    def toVector3r(self, wp):
        return airsim.Vector3r(wp[0], wp[1], wp[2])
if __name__ == "__main__":
    track_name = arg.track
    import contextlib
    learning_rate = 0.001
    learning_rate_change = 0.1
    learning_rate_change_epoch = 10
    batch_size = 32

    path_init_weight = ""
    loss_type = "MSE"
    configurations = []


    # with contextlib.closing(DaggerClient()) as sc:
    #     # generate random gate configurations within bounds set in config.json
    #     sc.generateGateConfigurations()
    #     configurations = deepcopy(sc.gateConfigurations)
    # rounds = 100
    # for i, gateConfig in enumerate(configurations):
    
         # for i in range(rounds):
        
        # if i==0:
        #     model_weight_path = arg.weight
        # else:
        #     model_weight_path = f'runs/ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_c={i-1}/best.pth'

        # model_weight_path = f'runs/ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_c={i-1}/best.pth'

    # beta = 0.9
    # for i in range(rounds):
    #     if i ==0 :
    #         os.system(f'python DaggerClient.py -w {path_init_weight}')
    #     else:
    #         model_weight_path = f'runs/ResNet32_ScaleV_body={batch_size}_lt={loss_type}_lr={learning_rate}_c={i-1}/best.pth'
    #         os.system(f'python DaggerClient.py -w {model_weight_path}')
   

        # model_weight_path = arg.weight

        # with contextlib.closing(DaggerClient(raceTrackName=f"track{i}",
        #                         beta = beta,
        #                         modelPath=model_weight_path,
        #                         device="cuda")) as sc:
        #     sc.gateConfigurations = [gateConfig]

        #     sc.loadNextGatePosition()

        #     # fly mission
        #     sc.gateMission(False, True)
        #     # sc.loadGatePositions([[3.555624961853027, 0.140624642372131, -0.65, -90.0]])
        #     sc.loadGatePositions(sc.config.gates['poses'])
        #     sc.reset()

        # os.system(f'python3 ../train_newloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X1Gate100 -r {i} -p {model_weight_path}')
        # exec(f'python3 ../train_newloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X1Gate100 -r {i} -p {model_weight_path}')
    
        # beta -= 0.05

    with contextlib.closing(DaggerClient(
            track_name,
            model_weight_path,
            beta= 0.9,
            # "/media/data2/teamICRA/runs/ResNet32_Loadall_scalewithVmax_250mix_world_body=32_lt=MSE_lr=0.001_c=run0/best.pth",
            device="cuda" )) as dc:
        dc.loadGatePositions([[3.555624961853027, 0.140624642372131, -0.65, -90.0]])
        dc.gateMission(True, True)
        # dc.close()
        dc.reset()
        