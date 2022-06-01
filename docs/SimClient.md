## SimClient 
This is the documentation for _'SimClient.py'_.  
This client can be used to generate datasets for droneracing with AirSim's SimpleFlight Multirotor mode. It contains the functionality to run a Mission.

`.gateConfigurations`:  
Type: `list(configurations)`  
where type of _configurations_: `list(gatePose)`  
where type of _gatePose_: `list(x,y,z,yaw)`  

`.currentGateConfiguration`:  
index into `SimClient.gateConfigurations`

`.timestep`: 1 / fps, time duration of single frame in seconds, based on framerate set in configuration

### Main
Creates instances of SimClient class with contextlib to make sure that the _close()_ function is called.  
  
First, random gate configurations are generated, constrained by the bounds set in 'config.json'.  
Afterwards, for each configuration a _SimClient_ instance is created with increasing track number, the gate configuration is set as the only element in `SimClient.gateConfigurations`.  
Further steps:
- `SimClient.loadNextGatePosition()`, set next gate configuration into airsim - so move gates to the poses in the configuration
- `SimClient.gateMission(false)`, fly mission through gates without showing trajectory markers
- `SimClient.loadGatePositions(initialPositions); SimClient.reset()`, reset simulation environment to initial state

### generate random gate configurations
The function `SimClient.generateGateConfigurations()` generates random gate pose configurations and saves them in `SimClient.gateConfigurations` as a list of configurations.  
> NOTE: the number of configurations set in _'config.json'_ can be higher than the actual value of distinct configurations generated.  

### gate mission
The function `SimClient.gateMission()` runs a full mission through the gates. 
- initialize velocity pid controller
- generate trajectory through gates
- save current configuration to dataset folder
- follow trajectory with sampled waypoints  

This function is used to generate a data set for a single track. It generates a trajectory through the current poses (center positions) of the gates in simulation and applies four PID controllers over the velocity command of the AirSim API to make the drone follow the trajectory. The trajectory can be visualized and the frames captured and saved in the data set folder. 