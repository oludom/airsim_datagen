
## AirSimController
Controller class, holding connections to airsim  
inherit from this class to create a client (as in SimClient.py). See [AirSimController.md](docs/AirSimController.md) for documentation.


_'SimClient.py'_ is a runable example that collects a stereo/depth data set with a drone flying in the Unreal Lab environment through a gate race track. See [SimClient.md](docs/SimClient.md) for documentation.  

The data collection can be configured with the file _'config.json'_. See [config.md](docs/config.md) for documentation.

## AirSim Settings
'settings.json' file has to be placed in ~/Documents/AirSim .  
use e.g. ln -s ~/dev/data-gen/settings.json ~/Documents/AirSim/settings.json to create symlink, such that 'settings.json' is always the same as the version in this repo.
