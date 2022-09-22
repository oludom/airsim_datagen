
import numpy as np
import matplotlib.pyplot as plt
import json


def read_file_tum(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [l[1:] for l in list if len(l) > 1]
    return np.array(list).astype(np.float)


def loadConfig(configPath):
    configFile = open(configPath, "r")
    class ConfigLoader:
        def __init__(self, **data):
            self.__dict__.update(data)

    data = json.load(configFile)
    return ConfigLoader(**data)


itypes = [
    "rgb",
    "rgbd",
    "d",
    "o",
    "do",
    "rgbdo",
    "rgbo"
]
basedir = "/data/datasets/dr_test"





# plot ground truth
# data = read_file_tum(basedir + f"/groundtruth_trajectory_track3.tum")
# # print(data.shape)
# xval = data[:, 0]
# yval = data[:, 1]
# plt.plot(xval, yval, label="min. snap")

# data = read_file_tum(basedir + f"/actual_trajectory_track3.tum")
# # print(data.shape)
# xval = data[:, 0]
# yval = data[:, 1]
# plt.plot(xval, yval, label="ground truth", color="black", linestyle="--", linewidth=2)

for i in [1, 3, 5, 6, 9]:
    track = f"track{i}"
# load config
    config = loadConfig(basedir + f"/{track}/config.json")

    poses = np.array(config.gates['poses'])

    xposes = poses[:, 0]
    yposes = poses[:, 1]
    plt.scatter(xposes, yposes, label="gates", color="black", marker="x", linewidths=2)

    data = read_file_tum(basedir + f"/{track}/actual_trajectory_{track}.tum")
    # print(data.shape)

    xval = data[:, 0]
    yval = data[:, 1]

    plt.plot(xval, yval, linewidth=0.8)



plt.grid()
# plt.legend()
plt.ylabel("position y axis world frame [m]")
plt.xlabel("position x axis world frame [m]")
plt.show()

