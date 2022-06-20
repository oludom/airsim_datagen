#!/usr/bin/env python

import sys

sys.path.append('../')

from datagen.DatasetLoader import RaceTracksDataset

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vectorDistanceLossFunction(preds, labels):
    pdist = torch.nn.PairwiseDistance()
    return pdist(preds, labels)


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

dataset_basepath = "/home/micha/dev/datasets/droneracing"
# dataset_basepath = "/data/datasets"
dataset_basename = "X4Gates_Circle_1"
# dataset_basename = "X4Gates_Circle_2"

dataset = RaceTracksDataset(dataset_basepath, dataset_basename)


train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)


# create figure
fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

wps = dataset.getWaypoints()
wps = torch.tensor(wps)
wps = wps[:20, :3]
wps = wps.tolist()

poses, velocities = dataset.getPoses(skip=0)

for v in velocities:
    print("yaw:", v[3])

poses = torch.tensor(poses)
poses = poses[:20, :3]
poses = poses.tolist()

velocities = velocities[:20]


for i in range(len(poses)):

    p = poses[i]
    v = velocities[i][:3].tolist()
    wp = wps[i]
    vlength = np.linalg.norm(v)
    print("len=", vlength)
    ax.scatter(*p, c=[0,0,0])  # plot the point (2,3,4) on the figure
    ax.scatter(*wp, c=[1,0,0])  # plot the point (2,3,4) on the figure

    ax.quiver(*p, *v, pivot='tail', length=vlength, arrow_length_ratio=0.03/vlength)

    print("yaw change request:", velocities[i][3])

plt.show()
