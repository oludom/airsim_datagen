## config json

> `debug`: boolean - show debug output  
> `framerate`: int - frame rate of data collection and pid controllers in simulation time, real-time frame rate can be lower  
> `waypoints_per_segment`: int - parameter for sampling function, how many waypoints will be generated between two segments of the trajectory, where a segment is the trajectory part between two gates  
> `gate_basename`: string - base name of the gate object in unreal engine  
> `dataset_basename`: string - name of data set folder (will be created if it does not exist)  
> `dataset_basepath`: string - path to data set parent directory  
> `gates`: dictionary - configuration of gates  
>> `poses`: list - list of type [x, y, z, yaw], center coordinates of gates in meters/degrees  
>> `range`: dictionary - contains ranges as [min, max] for axies, these ranges limit the random positioning of the gates to the gate center positions +- min/max range  
>> `grid_size`: int - step size for randomization of gate poses, so how many steps in ranges can be considered, including start and end of range  
>> `number_configurations`: maximum number of gate configurations generated with random poses within range