#!/usr/bin/env python

'''

simulation client for AirSimController.py 
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

from UnityPID import VelocityPID
from util import *

import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *
from nav_msgs.msg import Odometry
from std_msgs.msg import *


class ROSClient(AirSimInterface):

    def __init__(self, raceTrackName="track0", *args, **kwargs):

        # init super class (AirSimInterface)
        super().__init__(raceTrackName=raceTrackName,  *args, **kwargs)

        # do custom setup here
        if self.config.debug:
            self.c = curses.initscr()
            curses.noecho()
            curses.cbreak()

        self.gateConfigurations = []
        self.currentGateConfiguration = 0

        self.timestep = 1. / self.config.framerate

        
        # init ROS node
        self.mission_start = False

        print("Datagen ROS Client started")
        rospy.init_node('airsim_sim_node', anonymous=True)

        self.drone_pose_pub = rospy.Publisher('drone_pose', PoseStamped, queue_size = 1)  # publish drone pose
        self.odometry_pub = rospy.Publisher('/state_estimator/drone/odom_on_map', Odometry, queue_size = 1)

        self.gate_pose_array_pub = rospy.Publisher('gate_pose_array', PoseArray, queue_size = 1)  # publish gate poses

        self.control_input_sub = rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget, self.control_input_callback)

        # self.timer_odom = rospy.Timer(rospy.Duration(0.1), self.loopPublishOdomToROS) # this is a timer to output drone's odometry to ROS
        # self.timer_gate_pose = rospy.Timer(rospy.Duration(0.1), self.loopPublishGatePoseToROS) # this is a timer to output gate's pose to ROS

        # self.request_sub = rospy.Subscriber('/controller/request', Bool, self.request_callback)

    
    
    '''
    callback for collect odom and pose from the drone in the simulator and publish to ROS
    '''
    def loopPublishOdomToROS(self, event):
        print('Timer called at ' + str(event.current_real) )
        # if (self.mission_start == True):
        #     print("Publishing odom")

            # get pose from sim object and publish to ROS 
            # pose = self.client.simGetVehiclePose()
            # pose_msg = PoseStamped()
            # pose_msg.header.stamp = rospy.Time.now()
            # pose_msg.header.frame_id = "map"
            # pose_msg.pose.position.x = pose.position.x_val
            # pose_msg.pose.position.y = pose.position.y_val
            # pose_msg.pose.position.z = pose.position.z_val
            # pose_msg.pose.orientation.x = pose.orientation.x_val
            # pose_msg.pose.orientation.y = pose.orientation.y_val
            # pose_msg.pose.orientation.z = pose.orientation.z_val
            # pose_msg.pose.orientation.w = pose.orientation.w_val
            # self.drone_pose_pub.publish(pose_msg)

            # get odometry from sim object and publish to ROS 
            # odometry = self.client.simGetGroundTruthKinematics()
            # odometry = self.getOdometryGroundtruthUAV()
            # odometry_msg = Odometry()
            # odometry_msg.header.stamp = rospy.Time.now()
            # odometry_msg.header.frame_id = "map"
            # odometry_msg.pose.pose.position.x = odometry.position.x_val
            # odometry_msg.pose.pose.position.y = odometry.position.y_val
            # odometry_msg.pose.pose.position.z = odometry.position.z_val
            # odometry_msg.pose.pose.orientation.x = odometry.orientation.x_val
            # odometry_msg.pose.pose.orientation.y = odometry.orientation.y_val
            # odometry_msg.pose.pose.orientation.z = odometry.orientation.z_val
            # odometry_msg.pose.pose.orientation.w = odometry.orientation.w_val
            # odometry_msg.twist.twist.linear.x = odometry.linear_velocity.x_val
            # odometry_msg.twist.twist.linear.y = odometry.linear_velocity.y_val
            # odometry_msg.twist.twist.linear.z = odometry.linear_velocity.z_val
            # odometry_msg.twist.twist.angular.x = odometry.angular_velocity.x_val
            # odometry_msg.twist.twist.angular.y = odometry.angular_velocity.y_val
            # odometry_msg.twist.twist.angular.z = odometry.angular_velocity.z_val
            # self.odometry_pub.publish(odometry_msg)


    '''
    callback for publishing gate poses to ROS
    '''
    def loopPublishGatePoseToROS(self, event):

        # get gate poses from sim object and publish to ROS
        self.gates = self.client.simGetObjectPose("Gate")
        
        gate_pose_array_msg = PoseArray()
        gate_pose_array_msg.header.stamp = rospy.Time.now()
        gate_pose_array_msg.header.frame_id = "map"
        for gate in self.gates:
            gate_pose_array_msg.poses.append(Pose(position=Point(x=gate.x, y=gate.y, z=gate.z), orientation=Quaternion(x=0, y=0, z=0, w=1)))
        self.gate_pose_array_pub.publish(gate_pose_array_msg)


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
    This function takes body rates and thrust and use it to drive the drone
    This function is called every time step to update the simulation.
    '''
    def control_input_callback(self, msg):

        odometry = self.getOdometryGroundtruthUAV()
        odometry_msg = Odometry()
        odometry_msg.header.stamp = rospy.Time.now()
        odometry_msg.header.frame_id = "map"
        odometry_msg.pose.pose.position.x = odometry.position.x_val
        odometry_msg.pose.pose.position.y = odometry.position.y_val
        odometry_msg.pose.pose.position.z = odometry.position.z_val
        odometry_msg.pose.pose.orientation.x = odometry.orientation.x_val
        odometry_msg.pose.pose.orientation.y = odometry.orientation.y_val
        odometry_msg.pose.pose.orientation.z = odometry.orientation.z_val
        odometry_msg.pose.pose.orientation.w = odometry.orientation.w_val
        odometry_msg.twist.twist.linear.x = odometry.linear_velocity.x_val
        odometry_msg.twist.twist.linear.y = odometry.linear_velocity.y_val
        odometry_msg.twist.twist.linear.z = odometry.linear_velocity.z_val
        odometry_msg.twist.twist.angular.x = odometry.angular_velocity.x_val
        odometry_msg.twist.twist.angular.y = odometry.angular_velocity.y_val
        odometry_msg.twist.twist.angular.z = odometry.angular_velocity.z_val
        self.odometry_pub.publish(odometry_msg)

        if self.mission_start:
            self.client.moveByRollPitchYawrateThrottleAsync(msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust, duration = float (0.01))

        if self.config.debug:
            self.c.refresh()


    def close(self):

        self.client.simPause(False)

        if self.config.debug:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
        self.client.simFlushPersistentMarkers()

        # close super class (AirSimController)
        super().close()


    def angleDifference(self, a: float, b: float):
        return (a - b + 540) % 360 - 180



    '''
    generate a mission trajectory
    - generate trajectory through obtained gates
    - save current configuration to dataset folder
    - follow trajectory with sampled waypoints if PID controller is used

    options:
    - show_markers: boolean, if true, trajectory and drone current tracking pose will be visualized with red markers in simulation
    - keep_data: boolean, if true the sampled waypoints will be saved in the config dataset folder

    '''
    def generateGateMission(self, object_base_name = 'BP_AirLab2m1Gate_', num_object = 3, show_markers=False, keep_data = False):

        # get trajectory
        W_timed_waypoints, W_trajectory = self.generateTrajectoryFromObjectPositions(traj=True, object_base_name = object_base_name, num_object = num_object)
        path_in_world_frame, path_complete_in_world_frame = self.convertTrajectoryToWaypoints(W_timed_waypoints, W_trajectory,
                                                                    evaltime=self.config.roundtime)

        # show trajectory
        if show_markers:
            self.client.simPlotPoints(path_in_world_frame, color_rgba=[1.0, 0.0, 0.0, .2], size=10.0, duration=-1.0, is_persistent=True)
        
        # save current configuration and trajectory in data set folder
        if keep_data:
            data = {
                "waypoints": path_complete_in_world_frame
            }
            self.saveConfigToDataset(gateConfig, data)
        
        return path_complete_in_world_frame


    '''
    track a provided mission using PID controller
    - follow trajectory with the provided sampled waypoints list 
    - save images and JSON label to dataset folder

    options:
    - show_markers: boolean, if true, trajectory and drone current tracking pose will be visualized with red markers in simulation
    - captureImages: boolean, if true, each iteration will capture a frame of each camera, simulation is paused for this

    variable name prefix:
    W: coordinates in world frame
    B: coordinates in body frame (drone/uav)
    '''
    def trackTrajectoryPIDController(self, W_pathComplete, show_markers=False, captureImages=True):

        mission = True
        self.client.takeoffAsync().join() # take off
    
        time.sleep(3)   # make sure drone is not drifting anymore after takeoff
    
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

        lastWP = time.time()
        lastImage = time.time()
        lastIMU = time.time()
        lastPID = time.time()

        timePerWP = float(self.config.roundtime) / len(W_pathComplete)
        timePerImage = 1. / float(self.config.framerate)
        timePerIMU = 1. / float(self.config.imuRate)
        timePerPID = 1. / float(self.config.pidRate)

        cwpindex = 0
        cimageindex = 0
        if self.config.debug:
            self.c.clear()


        # controll loop
        while mission:

            # get and plot current waypoint (blue)
            wp = W_pathComplete[cwpindex]

            # show markers if applicable
            self.showMarkers(show_markers, wp)

            # get current time and time delta
            tn = time.time()

            nextWP = tn - lastWP > timePerWP
            nextImage = tn - lastImage > timePerImage
            nextIMU = tn - lastIMU > timePerIMU
            nextPID = tn - lastPID > timePerPID

            if show_markers:
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
                ByawGoal = self.angleDifference(wp[3], degrees(Wcstate[3]))
                # print(f"angle target: {ByawGoal:5.4f}")
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
            if nextWP and len(W_pathComplete) > (cwpindex + 1):
                cwpindex = cwpindex + 1
                lastWP = tn
            # end mission when no more waypoints available
            if len(W_pathComplete) <= (cwpindex + 1):
                mission = False

        if show_markers:
            # clear persistent markers
            self.client.simFlushPersistentMarkers()



    def showMarkers(self, show_markers, wp):
        if show_markers:
            self.client.simPlotPoints([airsim.Vector3r(wp[0], wp[1], wp[2])], color_rgba=[0.0, 0.0, 1.0, 1.0],
                                      size=10.0, duration=self.timestep, is_persistent=False)


    '''
    Set gate poses in simulation to provided configuration
    gates: [ [x, y, z, yaw], ...]
    ''' 
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



    # wp: [x, y, z]
    # returns airsim.Vector3r()
    def toVector3r(self, wp):
        return airsim.Vector3r(wp[0], wp[1], wp[2])



if __name__ == "__main__":

    ros_client = ROSClient()
    use_pid = True


    # fly mission
    sampled_wp_list_in_world_frame = ros_client.generateGateMission(object_base_name = 'BP_AirLab2m1Gate_', 
                                                            num_object = 3, show_markers=True, keep_data = False)
    if use_pid:
        ros_client.trackTrajectoryPIDController(sampled_wp_list_in_world_frame, show_markers=True, captureImages=False)
    else:
        print("Broadcasting trajectory to ROS")

    ros_client.reset()


    # import contextlib

    # configurations = []

    # with contextlib.closing(ROSClient()) as sc:
    #     # generate random gate configurations within bounds set in config.json
    #     sc.generateGateConfigurations()
    #     configurations = deepcopy(sc.gateConfigurations)



    # for i, gateConfig in enumerate(configurations):
    #     with contextlib.closing(ROSClient(raceTrackName=f"track{i}")) as sc:
    #         sc.gateConfigurations = [gateConfig]

    #         sc.loadNextGatePosition()

    #         # fly mission
    #         sampled_wp_list_in_world_frame = sc.generateGateMission(object_base_name = 'BP_AirLab2m1Gate_', 
    #                                                                 num_object = 3, show_markers=True, keep_data = False)
    #         if use_pid:
    #             sc.trackTrajectoryPIDController(sampled_wp_list_in_world_frame, show_markers=True, captureImages=False)
    #         else:
    #             print("Broadcasting trajectory to ROS")

    #         sc.loadGatePositions(sc.config.gates['poses'])
    #         sc.reset()