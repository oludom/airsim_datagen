
import numpy as np
import matplotlib.pyplot as plt


def read_file_tum(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [l[1:] for l in list if len(l) > 1]
    return np.array(list).astype(np.float)



itypes = [
    "rgb",
    "rgbd",
    "d",
    "o",
    "do",
    "rgbdo",
    "rgbo"
]
basedir = "/home/micha/dev/ml/orb_imitation/datagen/eval/trajectories/track3"

# plot ground truth
# data = read_file_tum(basedir + f"/groundtruth_trajectory_track3.tum")
# # print(data.shape)
# xval = data[:, 0]
# yval = data[:, 1]
# plt.plot(xval, yval, label="min. snap")

data = read_file_tum(basedir + f"/actual_trajectory_track3.tum")
# print(data.shape)
xval = data[:, 0]
yval = data[:, 1]
plt.plot(xval, yval, label="ground truth", color="black", linestyle="--", linewidth=2)

for el in itypes:

    data = read_file_tum(basedir + f"/trajectory_{el}.tum")
    # print(data.shape)

    xval = data[:, 0]
    yval = data[:, 1]

    plt.plot(xval, yval, label=el.upper())
plt.grid()
plt.legend()
plt.ylabel("position y axis world frame [m]")
plt.xlabel("position x axis world frame [m]")
plt.show()

